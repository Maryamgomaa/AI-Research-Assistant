import logging
from typing import List

import httpx

from .config import Config

log = logging.getLogger("web_search")


class SerpapiSearcher:
    """Optional public-source search via SerpAPI if the API key is configured."""

    SOURCE_CONFIG = {
        "google_scholar": {
            "engine": "google_scholar",
            "label": "Google Scholar",
        },
        "semantic_scholar": {
            "engine": "google",
            "site": "semanticscholar.org",
            "label": "Semantic Scholar",
        },
        "pubmed": {
            "engine": "google",
            "site": "pubmed.ncbi.nlm.nih.gov",
            "label": "PubMed",
        },
        "openreview": {
            "engine": "google",
            "site": "openreview.net",
            "label": "OpenReview",
        },
        "web": {
            "engine": "google",
            "label": "General Web",
        },
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or Config.SERPAPI_API_KEY

    async def search(
        self,
        query: str,
        num_results: int = 5,
        sources: list[str] | None = None,
        citation_threshold: int | None = None,
        sort_by: str = "submittedDate",
    ) -> list[dict]:
        if not self.api_key:
            log.warning("[SEARCH] SERPAPI_API_KEY is not configured. Skipping web search.")
            return []

        sources = sources or ["google_scholar"]
        results: list[dict] = []

        for source in sources:
            if source == "arxiv":
                continue

            config = self.SOURCE_CONFIG.get(source, self.SOURCE_CONFIG["web"])
            params = {
                "api_key": self.api_key,
                "engine": config["engine"],
                "q": query,
                "num": num_results,
            }

            if config.get("site"):
                params["q"] = f"{query} site:{config['site']}"

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get("https://serpapi.com/search.json", params=params)
                resp.raise_for_status()
                data = resp.json()

            source_label = config.get("label", source)
            raw_results = data.get("organic_results") or data.get("scholar_results") or []
            for item in raw_results:
                title = item.get("title") or item.get("result_title") or "Untitled"
                link = item.get("link") or item.get("result_url") or ""
                snippet = item.get("snippet") or item.get("snippet_text") or item.get("description") or ""
                citation_count = None
                if isinstance(item.get("inline_links"), dict):
                    citation_count = item.get("inline_links").get("cited_by") if item.get("inline_links") else None

                if citation_threshold is not None and citation_count is not None:
                    try:
                        citation_count_value = int(citation_count)
                        if citation_count_value < citation_threshold:
                            continue
                    except (ValueError, TypeError):
                        pass

                results.append({
                    "title": title,
                    "abstract": snippet,
                    "link": link,
                    "source": source_label,
                    "citation_count": citation_count,
                })

            log.info(f"[SEARCH] Found {len(raw_results)} web results for '{query}' from {source_label}")

        log.info(f"[SEARCH] Returning {len(results)} total web results for '{query}'")
        return results
