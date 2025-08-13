from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import uuid

app = FastAPI(
    title="CollabPlan-AI Core",
    version="0.1.0",
    description="Backend core for meeting transcription and action planning",
)

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
