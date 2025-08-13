from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

from sqlmodel import Session, select
from store.db import init_db, Run, Segment, engine

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
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    duration = 12.34  # placeholder seconds

    segs = [
        SegmentOut(idx=0, start=0.0, end=5.5, text="This is a placeholder transcript.", speaker="S1"),
        SegmentOut(idx=1, start=5.5, end=12.3, text="We will replace this with ASR output.", speaker="S2"),
    ]

    # Save to DB
    with Session(engine) as session:
        session.add(Run(id=run_id, meeting_date=req.meeting_date, source=req.source, duration_sec=duration))
        for s in segs:
            session.add(Segment(
                run_id=run_id, idx=s.idx, start=s.start, end=s.end,
                text=s.text, speaker=s.speaker
            ))
        session.commit()

    return TranscribeResponse(run_id=run_id, duration_sec=duration, segments=segs)