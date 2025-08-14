import os
import wave
import tempfile
import subprocess
from typing import List, Tuple

import webrtcvad

def ensure_mono16_wav(input_path: str) -> str:
    """
    Returns a temp mono 16 kHz WAV path for VAD.
    Uses ffmpeg if input is not already mono/16k WAV.
    """
    # Try to open as WAV and check format
    try:
        with wave.open(input_path, "rb") as w:
            ok = (
                w.getnchannels() == 1 and
                w.getsampwidth() == 2 and
                w.getframerate() == 16000
            )
        if ok:
            return input_path
    except wave.Error:
        pass  # not a WAV, convert

    # Convert with ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    out_path = tmp.name
    cmd = [
        "ffmpeg", "-nostats", "-loglevel", "error",
        "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-f", "wav", out_path
    ]
    try:
        subprocess.run(cmd, check=True)
    except Exception:
        # If conversion fails, clean up and rethrow
        if os.path.exists(out_path):
            os.unlink(out_path)
        raise
    return out_path


def _frames_10ms(pcm: bytes, sample_rate: int) -> List[bytes]:
    frame_bytes = int(0.01 * sample_rate) * 2  # 16-bit mono
    return [pcm[i:i+frame_bytes] for i in range(0, len(pcm), frame_bytes)]


def detect_voice_regions(
    wav_path: str,
    aggressiveness: int = 2,
    pad_ms: int = 300,
    min_region_ms: int = 400,
) -> List[Tuple[float, float]]:
    """
    Returns a list of (start_sec, end_sec) voice regions.
    - aggressiveness: 0..3, higher is more aggressive
    - pad_ms: soft padding around voiced frames
    - min_region_ms: drop tiny blips
    """
    vad = webrtcvad.Vad(aggressiveness)
    with wave.open(wav_path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        sr = w.getframerate()
        pcm = w.readframes(w.getnframes())

    frames = _frames_10ms(pcm, sr)
    voiced = [vad.is_speech(f, sr) for f in frames]

    # Pad voiced frames
    win = max(0, int(pad_ms / 10))
    padded = []
    for i in range(len(voiced)):
        lo = max(0, i - win)
        hi = min(len(voiced), i + win + 1)
        padded.append(any(voiced[lo:hi]))

    # Merge into regions
    regions: List[Tuple[float, float]] = []
    start = None
    for i, k in enumerate(padded):
        t = i * 0.01
        if k and start is None:
            start = t
        if not k and start is not None:
            regions.append((start, t))
            start = None
    if start is not None:
        regions.append((start, len(frames) * 0.01))

    # Drop very short segments and merge small gaps
    min_len = min_region_ms / 1000.0
    regions = [(s, e) for s, e in regions if e - s >= min_len]

    merged: List[Tuple[float, float]] = []
    for s, e in regions:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s - pe <= 0.15:  # merge gaps shorter than 150 ms
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged
