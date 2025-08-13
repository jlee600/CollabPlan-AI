from typing import List, Optional, Tuple
from dataclasses import dataclass

from faster_whisper import WhisperModel


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

    def transcribe_file(
        self,
        path: str,
        vad_filter: bool = True,
        beam_size: int = 5,
        language: Optional[str] = None,
    ) -> Tuple[float, List[ASRSegment]]:
        """
        Transcribe a single audio or video file.
        Returns: (duration_sec, segments)
        Notes:
        - faster-whisper can read many formats if ffmpeg is present.
        - It already segments speech, so a separate VAD is optional for MVP.
        """
        segments_iter, info = self.model.transcribe(
            path,
            vad_filter=vad_filter,
            beam_size=beam_size,
            language=language,  # None lets the model detect language
        )

        segments: List[ASRSegment] = []
        for i, seg in enumerate(segments_iter):
            # seg has .start, .end, .text
            segments.append(
                ASRSegment(
                    idx=i,
                    start=float(seg.start) if seg.start is not None else 0.0,
                    end=float(seg.end) if seg.end is not None else 0.0,
                    text=seg.text.strip(),
                    speaker=None,  # diarization is optional and can be added later
                )
            )

        duration_sec = float(info.duration) if info and info.duration else 0.0
        return duration_sec, segments


# Simple singleton holder so we do not reload the model per request
_asr_singleton: Optional[ASREngine] = None

def get_asr() -> ASREngine:
    global _asr_singleton
    if _asr_singleton is None:
        # Defaults are CPU friendly. You can change size to base or medium if your box is fast.
        _asr_singleton = ASREngine(model_size="small", device="cpu", compute_type="int8")
    return _asr_singleton
