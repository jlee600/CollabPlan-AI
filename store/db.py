from sqlmodel import SQLModel, Field, create_engine
from typing import Optional
from datetime import datetime, date

# SQLite 
DATABASE_URL = "sqlite:///./collabplan.db"
engine = create_engine(DATABASE_URL, echo=False)

class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)
    meeting_date: date
    source: str  # "upload", "recording", etc.
    duration_sec: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Segment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(foreign_key="run.id")
    idx: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None

def init_db():
    SQLModel.metadata.create_all(engine)
