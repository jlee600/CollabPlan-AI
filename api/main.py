from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
from sqlalchemy import func, desc
from sqlmodel import Session, select, delete
from store.db import init_db, Run, Segment, Plan, TaskRow, engine
from core.asr import get_asr, ASRSegment
from core.extract import extract_plan, LLMError
from core.diarize import diarize_file, assign_speakers_to_segments, build_speaker_name_map, apply_name_map
from starlette.responses import Response, StreamingResponse
import io, csv, json
import hashlib, mimetypes, pathlib, shutil, uuid, tempfile ,subprocess, os 
from fastapi.middleware.cors import CORSMiddleware

USE_VAD = os.getenv("USE_VAD", "1") == "1"  # set to 0 to disable quickly
VAD_AGGR = int(os.getenv("VAD_AGGR", "2"))

app = FastAPI(
    title="CollabPlan-AI Core",
    version="0.1.0",
    description="Backend core for meeting transcription and action planning",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

UPLOAD_DIR = pathlib.Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_AUDIO_CT = {
    "audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4", "audio/x-m4a",
    "audio/webm", "video/webm", "audio/ogg", "video/mp4",
    "audio/x-flac", "audio/flac"
}

def _safe_ext(filename: str, content_type: str) -> str:
    ext = pathlib.Path(filename).suffix.lower()
    if not ext:
        ext = mimetypes.guess_extension(content_type or "") or ".bin"
    return ext

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
    diarize: Optional[bool] = False

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

# ---------- run ----------

class RunBundle(BaseModel):
    run_id: str
    meeting_date: date
    duration_sec: Optional[float] = None
    segments: List[SegmentOut]
    summary: str
    tasks: List[Task]
    open_questions: List[str]

##############################
# ---------- Routes ----------
##############################

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe_audio(req: TranscribeRequest):
    """
    Real transcription using faster-whisper.
    - Reads the file at req.path
    - Saves segments to DB
    - Returns run_id, duration, segments
    """
    norm_path = _normalize_to_wav16k(req.path)
    # Run ASR
    asr = get_asr()
    duration, segs_asr = asr.transcribe_file(
        norm_path,
        use_ext_vad=USE_VAD,
        vad_aggr=VAD_AGGR,
        beam_size=5,
        language=None,
    )

    # Map ASR segments to API model
    segs_out = [
        SegmentOut(idx=s.idx, start=s.start, end=s.end, text=s.text, speaker=s.speaker)
        for s in segs_asr
    ]
    
    if req.diarize:
        try:
            turns = diarize_file(norm_path)
            segs_dicts = [dict(idx=s.idx, start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in segs_out]
            segs_with_spk = assign_speakers_to_segments(segs_dicts, turns)

            # NEW: map S0/S1... to names using intros
            name_map = build_speaker_name_map(segs_with_spk)
            if name_map:
                segs_with_spk = apply_name_map(segs_with_spk, name_map)

            segs_out = [SegmentOut(**s) for s in segs_with_spk]
        except Exception as e:
            print(f"[diarize] failed: {e}")

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

@app.post("/transcribe_upload", response_model=TranscribeResponse)
async def transcribe_upload(
    meeting_date: date = Form(...),
    diarize: bool = Form(False),
    file: UploadFile = File(...),
):
    """
    Accept an uploaded audio file, run ASR (and optional diarization), save to DB, return run info.
    """
    # persist upload to a temp file
    suffix = "." + (file.filename.split(".")[-1].lower() if file.filename and "." in file.filename else "bin")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
    
    norm_path = _normalize_to_wav16k(tmp_path)
    try:
        # ASR
        asr = get_asr()
        duration, segs_asr = asr.transcribe_file(
            norm_path,
            use_ext_vad=USE_VAD,
            vad_aggr=VAD_AGGR,
            beam_size=5,
            language=None,
        )

        segs_out = [
            SegmentOut(idx=s.idx, start=s.start, end=s.end, text=s.text, speaker=s.speaker)
            for s in segs_asr
        ]

        # diarization if requested
        if diarize:
            try:
                turns = diarize_file(norm_path)
                segs_dicts = [dict(idx=s.idx, start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in segs_out]
                segs_with_spk = assign_speakers_to_segments(segs_dicts, turns)

                name_map = build_speaker_name_map(segs_with_spk)
                if name_map:
                    segs_with_spk = apply_name_map(segs_with_spk, name_map)

                segs_out = [SegmentOut(**s) for s in segs_with_spk]
            except Exception as e:
                print(f"[diarize_upload] failed: {e}")

        # persist
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        with Session(engine) as session:
            session.add(Run(id=run_id, meeting_date=meeting_date, source="upload", duration_sec=duration))
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

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    text = (req.transcript or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty transcript")

    import re as _re
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    segs = []
    for i, s in enumerate(sentences):
        segs.append({"idx": i, "start": 0.0, "end": 0.0, "text": s, "speaker": None})

    # persist run + segments (duration unknown here)
    with Session(engine) as session:
        session.add(Run(id=run_id, meeting_date=req.meeting_date, source="text", duration_sec=None))
        for seg in segs:
            session.add(Segment(
                run_id=run_id,
                idx=seg["idx"],
                start=seg["start"],
                end=seg["end"],
                text=seg["text"],
                speaker=seg["speaker"],
            ))
        session.commit()

    # plan
    try:
        plan = extract_plan(req.meeting_date, segs)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=f"LLM failure: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze failed: {e}")

    with Session(engine) as session:
        # idempotency not needed (new run), just write
        session.add(Plan(
            run_id=run_id,
            summary=plan["summary"],
            open_questions_json=json.dumps(plan["open_questions"])
        ))

        for t in plan["tasks"]:
            # convert due_date "YYYY-MM-DD" -> date
            due_obj = None
            due_iso = t.get("due_date")
            if isinstance(due_iso, str) and due_iso:
                try:
                    due_obj = date.fromisoformat(due_iso)
                except ValueError:
                    due_obj = None

            session.add(TaskRow(
                run_id=run_id,
                title=t["title"],
                owner=t.get("owner"),
                due_date=due_obj,
                priority=t.get("priority"),
                dependencies_json=json.dumps(t.get("dependencies", [])),
                evidence_idx=None,
                evidence_span_json=None,
                confidence=t.get("confidence"),
            ))
        session.commit()

    # 6) response DTOs
    tasks_out = [
        Task(
            title=t["title"],
            owner=t.get("owner"),
            due_date=(date.fromisoformat(t["due_date"]) if isinstance(t.get("due_date"), str) else None),
            priority=t.get("priority"),
            dependencies=t.get("dependencies", []),
            evidence=Evidence(segment_idx=None, span=None),
            confidence=t.get("confidence"),
        )
        for t in plan["tasks"]
    ]

    return AnalyzeResponse(
        run_id=run_id,
        meeting_date=req.meeting_date,
        summary=plan["summary"],
        tasks=tasks_out,
        open_questions=plan["open_questions"],
    )

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

    # Persist plan and tasks
    with Session(engine) as session:
        # Remove old plan/tasks for idempotency if you re-run analyze
        session.exec(delete(TaskRow).where(TaskRow.run_id == run_id))
        session.exec(delete(Plan).where(Plan.run_id == run_id))

        session.add(Plan(
            run_id=run_id,
            summary=plan["summary"],
            open_questions_json=json.dumps(plan["open_questions"])
        ))

        for t in plan["tasks"]:
            due_iso = t.get("due_date")  # "YYYY-MM-DD" or None
            due_obj = date.fromisoformat(due_iso) if isinstance(due_iso, str) and due_iso else None

            session.add(TaskRow(
                run_id=run_id,
                title=t["title"],
                owner=t.get("owner"),
                due_date=due_obj,
                priority=t.get("priority"),
                dependencies_json=json.dumps(t.get("dependencies", [])),
                evidence_idx=None,
                evidence_span_json=None,
                confidence=t.get("confidence"),
            ))

        session.commit()

    return AnalyzePlanResponse(
        run_id=run_id,
        meeting_date=run.meeting_date,
        summary=plan["summary"],
        tasks=tasks,
        open_questions=plan["open_questions"],
    )

@app.post("/analyze_file", response_model=AnalyzeResponse)
def analyze_file(meeting_date: date = Form(...), file: UploadFile = File(...)):
    raw = file.file.read()

    def _text_bytes_to_str(b: bytes) -> str:
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("utf-8", errors="ignore")

    text = _text_bytes_to_str(raw)
    req = AnalyzeRequest(meeting_date=meeting_date, transcript=text)
    return analyze_text(req)

@app.get("/run/{run_id}", response_model=RunBundle)
def get_run(run_id: str):
    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.id == run_id)).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        segs = session.exec(
            select(Segment).where(Segment.run_id == run_id).order_by(Segment.idx)
        ).all()
        segs_out = [SegmentOut(idx=s.idx, start=s.start, end=s.end, text=s.text, speaker=s.speaker) for s in segs]

        plan = session.exec(select(Plan).where(Plan.run_id == run_id)).first()
        task_rows = session.exec(select(TaskRow).where(TaskRow.run_id == run_id)).all()

        summary = plan.summary if plan else ""
        open_q = json.loads(plan.open_questions_json) if plan else []

        tasks = []
        for tr in task_rows:
            tasks.append(Task(
                title=tr.title,
                owner=tr.owner,
                due_date=tr.due_date,
                priority=tr.priority,
                dependencies=json.loads(tr.dependencies_json or "[]"),
                evidence=Evidence(segment_idx=tr.evidence_idx, span=json.loads(tr.evidence_span_json) if tr.evidence_span_json else None),
                confidence=tr.confidence
            ))

    return RunBundle(
        run_id=run_id,
        meeting_date=run.meeting_date,
        duration_sec=run.duration_sec,
        segments=segs_out,
        summary=summary,
        tasks=tasks,
        open_questions=open_q
    )

@app.get("/runs")
def list_runs():
    with Session(engine) as session:
        runs = session.exec(select(Run).order_by(desc(Run.created_at))).all()
        items = []
        for r in runs:
            count_expr = func.count()  # pylint: disable=not-callable
            task_count = session.exec(
                select(count_expr).select_from(TaskRow).where(TaskRow.run_id == r.id)
            ).one()

            # small summary preview
            plan = session.exec(select(Plan).where(Plan.run_id == r.id)).first()
            preview = (plan.summary[:160] + "…") if plan and plan.summary and len(plan.summary) > 160 else (plan.summary if plan else "")
            items.append({
                "run_id": r.id,
                "meeting_date": r.meeting_date,
                "duration_sec": r.duration_sec,
                "summary_preview": preview,
                "task_count": int(task_count or 0),
            })
        return items

@app.get("/export/{run_id}.csv")
def export_csv(run_id: str):
    with Session(engine) as session:
        tasks = session.exec(select(TaskRow).where(TaskRow.run_id == run_id).order_by(TaskRow.id)).all()
        if tasks is None:
            raise HTTPException(status_code=404, detail=f"No tasks for run: {run_id}")

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["title", "owner", "due_date", "priority", "dependencies", "confidence"])
        for t in tasks:
            deps = json.loads(t.dependencies_json or "[]")
            writer.writerow([
                t.title or "",
                t.owner or "",
                t.due_date.isoformat() if t.due_date else "",
                t.priority or "",
                "; ".join(deps),
                ("" if t.confidence is None else f"{t.confidence:.2f}"),
            ])
        buf.seek(0)
        headers = {
            "Content-Disposition": f'attachment; filename="{run_id}.csv"'
        }
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)

@app.get("/export/{run_id}.md")
def export_markdown(run_id: str):
    with Session(engine) as session:
        run = session.exec(select(Run).where(Run.id == run_id)).first()
        if not run:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        plan = session.exec(select(Plan).where(Plan.run_id == run_id)).first()
        tasks = session.exec(select(TaskRow).where(TaskRow.run_id == run_id).order_by(TaskRow.id)).all()

        summary = plan.summary if plan else ""
        open_qs = json.loads(plan.open_questions_json or "[]") if plan else []

        lines = []
        lines.append(f"# Meeting Plan – {run_id}")
        lines.append(f"*Date:* {run.meeting_date.isoformat()}")
        lines.append("")
        lines.append("## Summary")
        lines.append(summary or "_(none)_")
        lines.append("")
        lines.append("## Tasks")
        if tasks:
            for t in tasks:
                deps = json.loads(t.dependencies_json or "[]")
                due = t.due_date.isoformat() if t.due_date else "—"
                conf = f"{t.confidence:.2f}" if t.confidence is not None else "—"
                lines.append(f"- **{t.title}** — owner: {t.owner or '—'}; due: {due}; priority: {t.priority or '—'}; deps: {', '.join(deps) or '—'}; conf: {conf}")
        else:
            lines.append("_(no tasks)_")
        lines.append("")
        lines.append("## Open Questions")
        if open_qs:
            for q in open_qs:
                lines.append(f"- {q}")
        else:
            lines.append("_(none)_")

        md = "\n".join(lines)
        return Response(content=md, media_type="text/markdown")
    
@app.post("/upload/audio")
def upload_audio(file: UploadFile = File(...)):
    ct = (file.content_type or "").lower()
    if ct not in _AUDIO_CT:
        # still allow, faster-whisper + ffmpeg can read many formats
        print(f"[upload_audio] unexpected content-type: {ct}")

    ext = _safe_ext(file.filename or "audio", ct)
    safe_name = f"{uuid.uuid4().hex}{ext}"
    out_path = UPLOAD_DIR / safe_name

    sha = hashlib.sha256()
    with out_path.open("wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)
            f.write(chunk)
    
    wav_name = f"{out_path.stem}.wav"
    wav_path = UPLOAD_DIR / wav_name
    try:
        tmp_wav = _normalize_to_wav16k(str(out_path))
        # if tmp_wav != wav_path, move/replace
        if tmp_wav != str(wav_path):
            shutil.move(tmp_wav, wav_path)
    except Exception as e:
        print(f"[upload_audio] normalize failed: {e}")
        wav_path = out_path

    return {
        "path": str(out_path),
        "sha256": sha.hexdigest(),
        "filename": file.filename,
        "content_type": ct,
        "bytes": out_path.stat().st_size,
    }

def _normalize_to_wav16k(src_path: str) -> str:
    """
    Convert any input audio to 16kHz mono WAV for consistent ASR + diarization.
    Returns a temp file path you should delete when done.
    """
    dst = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-ac", "1", "-ar", "16000",   # mono, 16k
                "-f", "wav", dst
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return dst
    except Exception as e:
        # if ffmpeg not present or conversion failed, fall back to original
        try:
            os.unlink(dst)
        except Exception:
            pass
        print(f"[normalize] ffmpeg convert failed: {e} (using original)")
        return src_path
    