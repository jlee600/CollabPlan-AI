from __future__ import annotations

import os
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple
from datetime import date

import requests
import dateparser


# Point to your running Ollama server
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


def _build_prompt(meeting_date: date, chunk_text: str) -> str:
    schema = textwrap.dedent("""
    Return ONLY valid JSON with this schema, no markdown and no extra text:
    {
      "summary": "string",
      "tasks": [
        {
          "title": "string",
          "owner": "string|null",
          "due_date_ref": "string|null",
          "dependencies": ["string"],
          "priority_hint": "High|Medium|Low|null",
          "evidence_text": "string"
        }
      ],
      "open_questions": ["string"]
    }
    """).strip()

    rules = textwrap.dedent(f"""
    You extract clear, actionable tasks with verbs.
    Meeting date: {meeting_date.isoformat()}.
    If owner is unclear, set owner to null.
    If a due date is spoken, put it in due_date_ref exactly as heard.
    If there are no tasks, return "tasks": [].
    Output must be JSON only.
    """).strip()

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


def _resolve_due_date(meeting_date: date, due_ref: Optional[str]) -> Tuple[Optional[str], float]:
    if not due_ref:
        return None, 0.0
    dt = dateparser.parse(due_ref, settings={"RELATIVE_BASE": meeting_date})
    if not dt:
        return None, 0.3
    return dt.date().isoformat(), 0.9


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

    merged_summary: List[str] = []
    merged_tasks: List[Dict[str, Any]] = []
    merged_open: List[str] = []

    for tx in texts:
        prompt = _build_prompt(meeting_date, tx)
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
