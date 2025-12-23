import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class GoogleCustomSearchWrapper:
    """
    Minimal wrapper for Google Custom Search JSON API.

    - No third-party dependencies (uses urllib from stdlib).
    - Requires:
        - api_key: Google API key
        - cse_id: Custom Search Engine ID ("cx")
    """

    def __init__(self, api_key: str, cse_id: str):
        if not api_key:
            raise ValueError("GoogleCustomSearchWrapper requires api_key")
        if not cse_id:
            raise ValueError("GoogleCustomSearchWrapper requires cse_id")
        self.api_key = api_key
        self.cse_id = cse_id

    def search(
        self,
        query: str,
        *,
        num: int = 5,
        safe: str = "active",
        timeout: float = 20.0,
    ) -> List[Dict[str, str]]:
        if not query or not str(query).strip():
            return []

        # Google API supports num in [1..10]
        try:
            num_i = int(num)
        except Exception:
            num_i = 5
        num_i = max(1, min(num_i, 10))

        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": str(query),
            "num": str(num_i),
            "safe": safe,
        }
        url = "https://www.googleapis.com/customsearch/v1?" + urllib.parse.urlencode(
            params
        )

        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL scheme: {url}")

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw) if raw else {}

        items = data.get("items") or []
        results: List[Dict[str, str]] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            results.append(
                {
                    "title": str(it.get("title", "") or ""),
                    "link": str(it.get("link", "") or ""),
                    "snippet": str(it.get("snippet", "") or ""),
                }
            )
        return results

    @staticmethod
    def to_text(results: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for i, r in enumerate(results or [], start=1):
            title = (r.get("title") or "").strip()
            snippet = (r.get("snippet") or "").strip()
            link = (r.get("link") or "").strip()
            lines.append(f"{i}. {title}\n{snippet}\n{link}".strip())
        return "\n\n".join(lines).strip()


