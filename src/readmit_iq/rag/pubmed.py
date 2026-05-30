"""
PubMed fetcher for the ReadmitIQ RAG corpus.

Uses NCBI E-utilities to search and download abstracts.
Caches results to JSON so repeated runs don't re-hit the API.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
Rate limit: 3 requests/sec without an API key.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from loguru import logger


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
REQUEST_DELAY_SEC = 0.4  # stay well under 3 req/sec
DEFAULT_BATCH_SIZE = 50  # how many PMIDs to fetch per efetch call


@dataclass(frozen=True)
class Abstract:
    """One PubMed abstract, normalized for downstream embedding."""

    pmid: str
    title: str
    abstract: str
    journal: str
    year: str
    authors: list[str]
    query: str  # which search query found this

    @property
    def text_for_embedding(self) -> str:
        """The text we hand to the embedder. Title + abstract."""
        return f"{self.title}\n\n{self.abstract}"


def _search_pmids(query: str, retmax: int) -> list[str]:
    """Step 1: search PubMed for PMIDs matching a query."""
    response = httpx.get(
        f"{EUTILS_BASE}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
        },
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["esearchresult"]["idlist"]


def _fetch_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    """Step 2: fetch full content for a batch of PMIDs."""
    if not pmids:
        return []
    response = httpx.get(
        f"{EUTILS_BASE}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return _parse_pubmed_xml(response.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed's efetch XML into a list of dicts."""
    root = ET.fromstring(xml_text)
    articles: list[dict[str, Any]] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or ""
        title = article.findtext(".//ArticleTitle") or ""
        # Abstract may be a single <AbstractText> or several with Label attrs.
        abstract_parts = []
        for at in article.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            text = (at.text or "").strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(p for p in abstract_parts if p)

        journal = article.findtext(".//Journal/Title") or ""
        year = article.findtext(".//PubDate/Year") or ""

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.findtext("LastName") or ""
            initials = author.findtext("Initials") or ""
            name = f"{last} {initials}".strip()
            if name:
                authors.append(name)

        if not abstract:
            continue  # skip articles without abstracts

        articles.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "journal": journal,
                "year": year,
                "authors": authors,
            }
        )
    return articles


def fetch_corpus(
    queries: list[str],
    output_path: Path,
    per_query: int = 50,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Abstract]:
    """
    Fetch a corpus of PubMed abstracts from a list of search queries.
    Writes results as JSON to output_path. Returns the parsed Abstracts.

    Deduplicates by PMID (an abstract found by multiple queries is recorded
    against its first query).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_pmids: set[str] = set()
    all_abstracts: list[Abstract] = []

    for query in queries:
        logger.info(f"Searching PubMed: {query!r}")
        pmids = _search_pmids(query, retmax=per_query)
        new_pmids = [p for p in pmids if p not in seen_pmids]
        logger.info(f"  Found {len(pmids)} PMIDs ({len(new_pmids)} new)")
        time.sleep(REQUEST_DELAY_SEC)

        # Fetch in batches to respect URL length limits.
        for i in range(0, len(new_pmids), batch_size):
            batch = new_pmids[i : i + batch_size]
            logger.info(f"  Fetching batch of {len(batch)} abstracts...")
            article_dicts = _fetch_abstracts(batch)
            for d in article_dicts:
                abstract = Abstract(query=query, **d)
                all_abstracts.append(abstract)
                seen_pmids.add(abstract.pmid)
            time.sleep(REQUEST_DELAY_SEC)

    logger.success(
        f"Fetched {len(all_abstracts)} abstracts across {len(queries)} queries"
    )

    # Persist as JSON.
    output_path.write_text(
        json.dumps([asdict(a) for a in all_abstracts], indent=2, ensure_ascii=False)
    )
    logger.success(f"Wrote corpus to {output_path}")
    return all_abstracts


if __name__ == "__main__":
    queries = [
        "heart failure 30-day readmission discharge",
        "pneumonia hospital readmission elderly",
        "COPD exacerbation discharge planning",
        "diabetes mellitus hospital discharge complications",
        "myocardial infarction post-discharge care",
        "stroke readmission rehabilitation",
        "sepsis recovery hospital discharge",
        "transitional care discharge planning",
        "medication reconciliation hospital discharge",
    ]
    fetch_corpus(
        queries=queries,
        output_path=Path("data/rag/abstracts.json"),
        per_query=50,
    )
