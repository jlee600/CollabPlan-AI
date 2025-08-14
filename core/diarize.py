from typing import List, Dict, Optional, Tuple
import os
import re
from pyannote.audio import Pipeline

_RE_INTRO = re.compile(
    r"\b(?:my name is|this is|i am|i'm)\s+([A-Z][a-z]{1,30})\b",
    flags=re.IGNORECASE,
)

def build_speaker_name_map(segments, max_scan=20):
    """
    Look at the first ~20 segments and learn who is S0/S1/etc.
    Returns dict like {"S0":"Tom", "S1":"Samir"}.
    """
    name_by_spk = {}
    seen_names = set()

    # also catch single-word name-only segments like 'Samir'
    def single_word_name(s: str) -> Optional[str]:
        s = s.strip()
        if " " in s or len(s) < 3:
            return None
        if s[0].isupper() and s[1:].islower():
            return s
        return None

    for seg in segments[:max_scan]:
        spk = seg.get("speaker")
        txt = seg.get("text", "") or ""
        if not spk or spk in name_by_spk:
            continue

        m = _RE_INTRO.search(txt)
        if m:
            name = m.group(1).strip().title()
            if name not in seen_names:
                name_by_spk[spk] = name
                seen_names.add(name)
                continue

        # fallback: single capitalized word as a standalone intro
        if len(txt.split()) == 1:
            guess = single_word_name(txt)
            if guess and guess not in seen_names:
                name_by_spk[spk] = guess
                seen_names.add(guess)

    return name_by_spk


def apply_name_map(segments, name_map):
    """
    Replace speaker IDs with names where known.
    """
    out = []
    for seg in segments:
        spk = seg.get("speaker")
        label = name_map.get(spk, spk) if spk else None
        out.append({**seg, "speaker": label})
    return out

# Cache the pipeline so we do not re-load per request
_diar_pipeline: Optional[Pipeline] = None

def get_diar_pipeline() -> Pipeline:
    global _diar_pipeline
    if _diar_pipeline is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError("HUGGINGFACE_TOKEN env var not set. Get a free token at huggingface.co and export it.")
        # This pipeline does VAD + segmentation + clustering
        _diar_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
    return _diar_pipeline


def diarize_file(path: str) -> List[Dict]:
    """
    Returns a list of diarization turns:
    [{"speaker": "S0", "start": float_sec, "end": float_sec}]
    """
    pipeline = get_diar_pipeline()
    diar = pipeline(path)
    # Map speaker labels to S0, S1, ...
    # pyannote labels: "SPEAKER_00"
    mapping = {}
    turns: List[Dict] = []
    next_id = 0

    for speech_turn, _, speaker in diar.itertracks(yield_label=True):
        if speaker not in mapping:
            mapping[speaker] = f"S{next_id}"
            next_id += 1
        spk = mapping[speaker]
        turns.append({
            "speaker": spk,
            "start": float(speech_turn.start),
            "end": float(speech_turn.end),
        })

    # Sort by start time
    turns.sort(key=lambda x: x["start"])
    return turns


def assign_speakers_to_segments(
    segments: List[Dict],
    turns: List[Dict],
    min_overlap: float = 0.2
) -> List[Dict]:
    """
    For each ASR segment with [start,end], assign the speaker with max time overlap.
    If max overlap < min_overlap seconds, keep speaker as None.
    """
    out = []
    for seg in segments:
        s0, e0 = float(seg["start"]), float(seg["end"])
        best_spk = None
        best_olap = 0.0
        for tr in turns:
            s1, e1 = tr["start"], tr["end"]
            # overlap
            start = max(s0, s1)
            end = min(e0, e1)
            olap = max(0.0, end - start)
            if olap > best_olap:
                best_olap = olap
                best_spk = tr["speaker"]
        if best_spk is not None and best_olap >= min_overlap:
            seg = {**seg, "speaker": best_spk}
        out.append(seg)
    return out
