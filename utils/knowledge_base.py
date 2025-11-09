import os
import json
import time
import urllib.parse
from typing import Optional, Dict, Any

import requests


DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "knowledge.json")


def _ensure_db() -> None:
    if not os.path.isdir(DB_DIR):
        try:
            os.makedirs(DB_DIR, exist_ok=True)
        except Exception:
            pass
    if not os.path.isfile(DB_PATH):
        try:
            with open(DB_PATH, "w", encoding="utf-8") as f:
                json.dump({"terms": {}}, f)
        except Exception:
            pass


def _load() -> Dict[str, Any]:
    _ensure_db()
    try:
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"terms": {}}


def _save(db: Dict[str, Any]) -> None:
    try:
        with open(DB_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_term(term: str) -> Optional[Dict[str, Any]]:
    db = _load()
    return db.get("terms", {}).get(term.lower())


def add_term(term: str, summary: str, source_url: str) -> None:
    db = _load()
    terms = db.setdefault("terms", {})
    terms[term.lower()] = {
        "term": term,
        "summary": summary,
        "source": source_url,
        "learned_at": int(time.time())
    }
    _save(db)


def learn_from_web(term: str) -> Optional[Dict[str, str]]:
    """Try to learn a term from the web (Wikipedia first, then DuckDuckGo).

    Returns {"summary": str, "source": str} or None.
    """
    q = term.strip()
    if not q:
        return None

    # 1) Wikipedia REST API summary
    try:
        title = urllib.parse.quote(q.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
        r = requests.get(url, timeout=4, headers={"User-Agent": "blind-nav/1.0"})
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract")
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title}")
            if extract:
                return {"summary": extract, "source": page_url}
    except Exception:
        pass

    # 2) DuckDuckGo Instant Answer API
    try:
        url = "https://api.duckduckgo.com/"
        r = requests.get(url, params={"q": q, "format": "json", "no_redirect": 1, "no_html": 1}, timeout=4)
        if r.status_code == 200:
            data = r.json()
            abstract = (data.get("AbstractText") or "").strip()
            source_url = (data.get("AbstractURL") or data.get("Redirect") or "").strip()
            if abstract:
                return {"summary": abstract, "source": source_url or f"https://duckduckgo.com/?q={urllib.parse.quote(q)}"}
    except Exception:
        pass

    # 3) Fallback: store the search URL only
    try:
        return {
            "summary": "I couldn't find a reliable summary, but I saved a search link.",
            "source": f"https://www.google.com/search?q={urllib.parse.quote(q)}"
        }
    except Exception:
        return None


