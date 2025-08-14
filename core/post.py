from __future__ import annotations
from typing import Iterable, List, Optional
from datetime import datetime, date
import json

INTERROGATIVES = {"who", "what", "when", "where", "why", "how", "which"}

def canonicalize_owner(owner, intro_names: set[str]) -> Optional[List[str]]:
    """
    Accepts a string like 'Will|Sameer' or a list.
    Returns a de-duplicated list with simple fixes and snapping to intro names.
    """
    if not owner:
        return None

    if isinstance(owner, str):
        parts = [p.strip() for p in owner.replace("|", ",").split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in owner if str(p).strip()]

    fixes = {"samir": "Sameer"}  # add more if needed
    out: List[str] = []
    seen = set()
    for p in parts:
        low = p.lower()
        p = fixes.get(low, p)
        # snap to an intro name if case differs
        for cand in intro_names:
            if low == cand.lower():
                p = cand
                break
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out or None

def clean_dependencies(dep_list: Iterable[str] | None, task_titles: Iterable[str]) -> List[str]:
    """Keep only deps that exactly match other task titles. Preserve order, drop dups."""
    if not dep_list:
        return []
    title_set = set(t for t in task_titles if t)
    out: List[str] = []
    seen = set()
    for d in dep_list:
        if d in title_set and d not in seen:
            out.append(d)
            seen.add(d)
    return out

def is_question(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    first = s.split(" ", 1)[0].lower()
    return s.endswith("?") or first in INTERROGATIVES

def looks_truncated(s: str) -> bool:
    """Very cheap truncation check."""
    if not s:
        return False
    return not s[-1] in ".!?"

def task_confidence(t: dict) -> float:
    """
    Simple score from signals. Range about 0.3 to 0.95.
    """
    score = 0.3
    if t.get("due_date"):
        score += 0.3
    if t.get("owner"):
        score += 0.2
    title = (t.get("title") or "").lower()
    if any(v in title for v in ["ship", "create", "draft", "review", "prepare", "deploy", "document"]):
        score += 0.1
    if t.get("evidence_text"):
        score += 0.1
    return min(0.95, score)

def intro_name_set(segments: list[dict]) -> set[str]:
    """
    Collect names introduced early. Works with diarized speaker field as well.
    """
    names: set[str] = set()
    for s in segments[:20]:
        spk = s.get("speaker")
        if isinstance(spk, str) and spk:
            names.add(spk)
    return names
