from typing import List, Optional, Tuple
from dataclasses import dataclass
import os
import tempfile
import subprocess

from faster_whisper import WhisperModel
from core.vad import ensure_mono16_wav, detect_voice_regions

@dataclass
class ASRSegment:
    idx: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class ASREngine:
    """
    Thin wrapper around faster-whisper.
    Loads the model once and exposes a transcribe() method.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
        """
        model_size: tiny, base, small, medium, large-v2, etc.
        device: cpu or cuda
        compute_type: int8, int8_float16, float16, float32
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def _transcribe_one_file(
        self,
        path: str,
        beam_size: int = 5,
        language: Optional[str] = None,
        vad_filter: bool = False,   # we set False when using external VAD
    ) -> Tuple[float, List[ASRSegment]]:
        segments_iter, info = self.model.transcribe(
            path,
            vad_filter=vad_filter,
            beam_size=beam_size,
            language=language,
            temperature=0.0,
            best_of=1,
        )
        out: List[ASRSegment] = []
        for i, seg in enumerate(segments_iter):
            out.append(
                ASRSegment(
                    idx=i,
                    start=float(seg.start) if seg.start is not None else 0.0,
                    end=float(seg.end) if seg.end is not None else 0.0,
                    text=seg.text.strip(),
                    speaker=None,
                )
            )
        duration_sec = float(info.duration) if info and info.duration else (out[-1].end if out else 0.0)
        return duration_sec, out

    def transcribe_file(
        self,
        path: str,
        *,
        use_ext_vad: bool = True,
        vad_aggr: int = 2,
        beam_size: int = 5,
        language: Optional[str] = None,
    ) -> Tuple[float, List[ASRSegment]]:
        """
        Transcribe a single audio or video file.
        Returns: (duration_sec, segments)
        If use_ext_vad is True, run WebRTC VAD first, then ASR only on speech regions.
        """
        if not use_ext_vad:
            # Current behavior, keep internal VAD if you want it
            return self._transcribe_one_file(path, beam_size=beam_size, language=language, vad_filter=True)

        tmp_wav = None
        try:
            # 1) Prepare a clean mono 16k wav for VAD
            wav_path = ensure_mono16_wav(path)
            if wav_path != path:
                tmp_wav = wav_path

            # 2) Detect speech regions
            regions = detect_voice_regions(wav_path, aggressiveness=vad_aggr)
            if not regions:
                # Fallback to plain ASR on the full file
                return self._transcribe_one_file(path, beam_size=beam_size, language=language, vad_filter=False)

            all_segments: List[ASRSegment] = []
            idx = 0
            max_end = 0.0

            # 3) Slice and transcribe per region
            for rs, re in regions:
                max_end = max(max_end, re)
                slice_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                slice_tmp.close()
                slice_path = slice_tmp.name

                # ffmpeg slice
                subprocess.run(
                    [
                        "ffmpeg", "-nostats", "-loglevel", "error",
                        "-y", "-ss", f"{rs:.2f}", "-to", f"{re:.2f}",
                        "-i", wav_path, "-ac", "1", "-ar", "16000", slice_path
                    ],
                    check=True,
                )

                _, segs = self._transcribe_one_file(slice_path, beam_size=beam_size, language=language, vad_filter=False)
                os.unlink(slice_path)

                # offset timings, reindex
                for s in segs:
                    s.start = rs + s.start
                    s.end = rs + s.end
                    s.idx = idx
                    idx += 1
                    all_segments.append(s)

            duration_sec = max_end if all_segments else 0.0
            return duration_sec, all_segments

        except Exception:
            # Any issue with VAD or slicing falls back to whole-file ASR
            return self._transcribe_one_file(path, beam_size=beam_size, language=language, vad_filter=False)
        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                os.unlink(tmp_wav)


# Simple singleton holder so we do not reload the model per request
_asr_singleton: Optional[ASREngine] = None

def get_asr() -> ASREngine:
    global _asr_singleton
    if _asr_singleton is None:
        # Defaults are CPU friendly. You can change size to base or medium if your box is fast.
        _asr_singleton = ASREngine(model_size="small", device="cpu", compute_type="int8")
    return _asr_singleton
