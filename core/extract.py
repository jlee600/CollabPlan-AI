from __future__ import annotations
import re
import os
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from datetime import date, datetime, timedelta, time
import requests
import dateparser
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None

_PERSON_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
# Point to running Ollama server
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
DEFAULT_MODEL = os.getenv("ANALYZER_MODEL", "mistral:7b")

class LLMError(RuntimeError):
    pass


def _ollama_generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.1, request_json: bool = True) -> str:
    """
    One-shot call to Ollama. If request_json=True, we add format:'json'.
    If that yields an empty response, retry once without the format hint.
    Any Ollama 'error' comes back as LLMError so the API can return 502.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }
    if request_json:
        payload["format"] = "json"

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, dict) and data.get("error"):
        raise LLMError(f"Ollama error: {data['error']}")

    text = (data.get("response") or "").strip()

    # If empty and we asked for JSON, retry once without the JSON hint
    if not text and request_json:
        return _ollama_generate(prompt, model=model, temperature=temperature, request_json=False)

    if os.getenv("ANALYZER_DEBUG") == "1":
        print("\n--- OLLAMA RAW START ---\n", text[:1200], "\n--- OLLAMA RAW END ---\n")

    if not text:
        raise LLMError("Empty response from model.")
    return text


def _build_prompt(meeting_date: date, chunk_text: str, speakers: List[str]) -> str:
    speaker_list = ", ".join(speakers) if speakers else "unknown"

    schema = """
    Return ONLY valid JSON with this schema, no markdown and no extra text:
    {
      "summary": "string",        // write 5-7 sentences: context, named speakers and roles, concrete points, status, next steps
      "tasks": [
        {
          "title": "string",
          "owner": "string|null",          // choose from the known speakers if possible: EXACT NAME
          "due_date_ref": "string|null",   // e.g., "next Friday", "EOM", "beta soon"
          "dependencies": ["string"],
          "priority_hint": "High|Medium|Low|null",
          "evidence_text": "string"        // short quote from transcript supporting this task
        }
      ],
      "open_questions": ["string"]
    }
    """.strip()

    rules = f"""
    Meeting date: {meeting_date.isoformat()}.
    Known speakers in this transcript: {speaker_list}.
    When setting "owner", prefer an exact match from the known speakers.
    Only create tasks that are clearly implied by the text.
    Always fill "evidence_text" with a short quote backing the task.
    Keep "priority_hint" realistic. Use High only if urgency words are present (urgent, asap, by EOD/EOW, blocker).
    If a due date is spoken, put it in due_date_ref exactly as heard.
    Output must be JSON only.
    """.strip()

    return f"{rules}\n\n{schema}\n\nTranscript:\n\"\"\"\n{chunk_text}\n\"\"\""


def _parse_json_or_empty(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        return {"summary": "", "tasks": [], "open_questions": []}

def chunk_segments_to_text(segments: List[Dict[str, Any]], max_chars: int = 3000) -> List[str]:
    chunks, cur, cur_len = [], [], 0
    for seg in segments:
        t = seg.get("text") or ""
        if not t:
            continue
        if cur_len + len(t) > max_chars and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(t)
        cur_len += len(t) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def extract_plan(meeting_date: date, segments: List[Dict[str, Any]], model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    texts = chunk_segments_to_text(segments)
    speakers = sorted({(s.get("speaker") or "").strip() for s in segments if s.get("speaker")})

    merged_summary: List[str] = []
    merged_tasks: List[Dict[str, Any]] = []
    merged_open: List[str] = []

    for tx in texts:
        prompt = _build_prompt(meeting_date, tx, speakers)
        raw = _ollama_generate(prompt, model=model, temperature=0.1, request_json=True)
        obj = _parse_json_or_empty(raw)

        if obj.get("summary"):
            merged_summary.append(obj["summary"])

        for t in obj.get("tasks", []):
            title = (t.get("title") or "").strip()
            if not title:
                continue
            owner = t.get("owner")
            due_ref = t.get("due_date_ref")
            due_iso, due_conf = _resolve_due_date(meeting_date, due_ref)
            deps = t.get("dependencies") or []
            phint = t.get("priority_hint")
            ev = t.get("evidence_text") or ""

            merged_tasks.append({
                "title": title,
                "owner": owner,
                "due_date": due_iso,
                "priority": phint,
                "dependencies": deps,
                "evidence_text": ev,
                "confidence": due_conf
            })

        for q in obj.get("open_questions", []):
            if isinstance(q, str) and q.strip():
                merged_open.append(q.strip())
    
    # Collect names from whole transcript and fill missing owners if the evidence mentions them
    full_text = " ".join([s.get("text", "") for s in segments])
    name_candidates = _extract_person_names(full_text)

    for t in merged_tasks:
        if t.get("owner"):
            continue
        ev = t.get("evidence_text", "") or t.get("title", "")
        nm = _nearest_name_in_text(ev, name_candidates)
        if nm:
            t["owner"] = nm
            t["confidence"] = max(t.get("confidence", 0.0), 0.6)

    # Merge summary and dedupe tasks by lowercase title
    final_summary = " ".join(merged_summary)[:600] if merged_summary else ""
    seen = set()
    deduped_tasks: List[Dict[str, Any]] = []
    for t in merged_tasks:
        key = t["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_tasks.append(t)

    # Minimal fallback so analyze never returns empty
    if not final_summary and not deduped_tasks:
        flat = " ".join([s.get("text", "") for s in segments])[:400]
        deduped_tasks.append({
            "title": "Review transcript and extract tasks",
            "owner": None,
            "due_date": None,
            "priority": None,
            "dependencies": [],
            "evidence_text": flat[:180],
            "confidence": 0.2
        })
        final_summary = "Auto fallback: transcript unclear or too short."

    return {
        "summary": final_summary,
        "tasks": deduped_tasks,
        "open_questions": merged_open[:10]
    }

def _extract_person_names(text: str) -> List[str]:
    """
    Return a small unique list of person-like names.
    Uses spaCy PERSON entities if available, else simple regex on capitalized words.
    """
    names: List[str] = []
    if _NLP:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                n = ent.text.strip()
                if 2 <= len(n) <= 40 and n not in names:
                    names.append(n)
    else:
        # very simple fallback
        for m in _PERSON_RE.finditer(text):
            n = m.group(1)
            if n not in names:
                names.append(n)
    # keep it short
    return names[:10]


def _nearest_name_in_text(evidence: str, names: List[str]) -> Optional[str]:
    """Pick a name that appears inside the evidence_text, if any."""
    ev = evidence or ""
    for n in names:
        # simple contains, case sensitive first, then case fold
        if n in ev or n.lower() in ev.lower():
            return n
    return None

# ---------- due date ----------

def _last_day_of_month(d: date) -> date:
    if d.month == 12:
        return date(d.year, 12, 31)
    first_next = date(d.year, d.month + 1, 1)
    return first_next - timedelta(days=1)


def _end_of_week(d: date) -> date:
    # treat Friday as end of week
    # weekday(): Mon=0..Sun=6
    delta = (4 - d.weekday()) % 7
    return d + timedelta(days=delta)

def _next_weekday(d: date, target_idx: int) -> date:
    """Next occurrence of weekday >= tomorrow (never today)."""
    days_ahead = (target_idx - d.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return d + timedelta(days=days_ahead)

def _this_or_next_weekday(d: date, target_idx: int) -> date:
    """
    'this Friday' => upcoming Friday in the current week (could be today).
    If the day has passed this week, use the one in the coming week.
    """
    days_ahead = (target_idx - d.weekday()) % 7
    return d + timedelta(days=days_ahead)

def _strip_leading_by(s: str) -> str:
    return re.sub(r"^\s*(by|on|at)\s+", "", s, flags=re.IGNORECASE)

def _rel_base_dt(meeting_date: date) -> datetime:
    # dateparser requires a datetime for RELATIVE_BASE
    return datetime.combine(meeting_date, datetime.min.time())

def _resolve_due_date(meeting_date: date, due_ref: Optional[str]) -> Tuple[Optional[str], float]:
    """
    Extend previous resolver:
    - EOD / COB: same day
    - EOW: that week's Friday
    - EOM: last day of month
    Otherwise, let dateparser try.
    """
    if not due_ref:
        return None, 0.0
    s_raw = due_ref.strip()
    if not s_raw:
        return None, 0.0
    s = due_ref.strip().lower()

    # Short forms
    if s in {"eod", "cob", "end of day", "close of business"}:
        return meeting_date.isoformat(), 0.8
    if s in {"eow", "end of week"}:
        return _end_of_week(meeting_date).isoformat(), 0.8
    if s in {"eom", "end of month"}:
        return _last_day_of_month(meeting_date).isoformat(), 0.8

    # Phrases like "by EOD/EOW/EOM"
    if "by eod" in s or "by cob" in s or "by close of business" in s:
        return meeting_date.isoformat(), 0.8
    if "by eow" in s or "by end of week" in s:
        return _end_of_week(meeting_date).isoformat(), 0.8
    if "by eom" in s or "by end of month" in s:
        return _last_day_of_month(meeting_date).isoformat(), 0.8
    
    wd_map = {
        "monday": 0, "mon": 0,
        "tuesday": 1, "tue": 1, "tues": 1,
        "wednesday": 2, "wed": 2,
        "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
        "friday": 4, "fri": 4,
        "saturday": 5, "sat": 5,
        "sunday": 6, "sun": 6,
    }
    
    import re as _re
    m = _re.search(r"\b(this|next)?\s*(monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|thurs|friday|fri|saturday|sat|sunday|sun)\b", s)
    if m:
        which = (m.group(1) or "").strip()
        target_idx = wd_map[m.group(2)]
        if which == "next":
            # strictly the following week
            dt = _next_weekday(meeting_date, target_idx)
        else:
            # "this <weekday>" or bare weekday => this week (today allowed)
            dt = _this_or_next_weekday(meeting_date, target_idx)
        return dt.isoformat(), 0.88

    # "in N days/weeks"
    m = _re.search(r"\b(?:in|within)\s+(\d+)\s+(day|days|week|weeks)\b", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        dt = meeting_date + (timedelta(weeks=n) if "week" in unit else timedelta(days=n))
        return dt.isoformat(), 0.88

    # fallback to dateparser (datetime)
    try:
        dt = dateparser.parse(s_raw, settings={"RELATIVE_BASE": _rel_base_dt(meeting_date)})
        if dt:
            return dt.date().isoformat(), 0.9
    except Exception:
        pass

    return None, 0.3