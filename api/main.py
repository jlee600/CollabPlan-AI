from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

from sqlmodel import Session, select
from store.db import init_db, Run, Segment, engine

from core.asr import get_asr, ASRSegment
from core.extract import extract_plan, LLMError

import uuid

app = FastAPI(
    title="CollabPlan-AI Core",
    version="0.1.0",
    description="Backend core for meeting transcription and action planning",
)

init_db()

# ---------- analyze ----------

class Evidence(BaseModel):
    segment_idx: Optional[int] = Field(None, description="Index of transcript segment")
    span: Optional[List[int]] = Field(None, description="[start_char, end_char] in that segment")

class Task(BaseModel):
    title: str
    owner: Optional[str] = None
    due_date: Optional[date] = None
    priority: Optional[str] = Field(None, description="High, Medium, Low")
    dependencies: List[str] = []
    evidence: Optional[Evidence] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class AnalyzeRequest(BaseModel):
    meeting_date: date
    transcript: str

class AnalyzeResponse(BaseModel):
    run_id: str
    meeting_date: date
    summary: str
    tasks: List[Task]
    open_questions: List[str]

# ---------- transcribe ----------

class SegmentOut(BaseModel):
    idx: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

class TranscribeRequest(BaseModel):
    meeting_date: date
    source: str  # "upload", "recording"
    path: str    # local path to audio file

class TranscribeResponse(BaseModel):
    run_id: str
    duration_sec: float
    segments: List[SegmentOut]

# ---------- analyze/runID ----------

class AnalyzePlanResponse(BaseModel):
    run_id: str
    meeting_date: date
    summary: str
    tasks: List[Task]
    open_questions: List[str]

# ---------- Routes ----------

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):    
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    summary = "Placeholder summary. Pipeline not implemented yet."
    tasks: List[Task] = []
    open_questions: List[str] = []

    return AnalyzeResponse(
        run_id=run_id,
        meeting_date=req.meeting_date,
        summary=summary,
        tasks=tasks,
        open_questions=open_questions,
    )

@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_audio(req: TranscribeRequest):
    """
    Real transcription using faster-whisper.
    - Reads the file at req.path
    - Saves segments to DB
    - Returns run_id, duration, segments
    """
    # Run ASR
    asr = get_asr()
    duration, segs_asr = asr.transcribe_file(req.path)

    # Map ASR segments to API model
    segs_out = [
        SegmentOut(idx=s.idx, start=s.start, end=s.end, text=s.text, speaker=s.speaker)
        for s in segs_asr
    ]

    # Persist
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    with Session(engine) as session:
        session.add(Run(id=run_id, meeting_date=req.meeting_date, source=req.source, duration_sec=duration))
        for s in segs_out:
            session.add(Segment(
                run_id=run_id,
                idx=s.idx,
                start=s.start,
                end=s.end,
                text=s.text,
                speaker=s.speaker
            ))
        session.commit()

    return TranscribeResponse(run_id=run_id, duration_sec=duration, segments=segs_out)

@app.post("/analyze/{run_id}", response_model=AnalyzePlanResponse)
def analyze_run(run_id: str):
    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.id == run_id)).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        segs = session.exec(select(Segment).where(Segment.run_id == run_id).order_by(Segment.idx)).all()
        seg_dicts = [{"idx": s.idx, "start": s.start, "end": s.end, "text": s.text, "speaker": s.speaker} for s in segs]

    try:
        plan = extract_plan(run.meeting_date, seg_dicts)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM failure: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {e}")

    tasks = [
        Task(
            title=t["title"],
            owner=t.get("owner"),
            due_date=t.get("due_date"),
            priority=t.get("priority"),
            dependencies=t.get("dependencies", []),
            evidence=Evidence(segment_idx=None, span=None),
            confidence=t.get("confidence"),
        )
        for t in plan["tasks"]
    ]

    return AnalyzePlanResponse(
        run_id=run_id,
        meeting_date=run.meeting_date,
        summary=plan["summary"],
        tasks=tasks,
        open_questions=plan["open_questions"],
    )
