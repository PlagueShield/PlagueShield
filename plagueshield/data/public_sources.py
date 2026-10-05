"""Fetchers for genuinely public plague data.

What is and is not publicly available, stated plainly because it shapes the
whole system:

  * **Isolate data: genuinely public.** NCBI Pathogen Detection holds
    Y. pestis genomes with standardised AMR genotype calls from AMRFinderPlus,
    queryable without credentials. This is real data about real isolates and
    is what calibrates the resistance prior.

  * **Outbreak data: public in aggregate.** WHO Disease Outbreak News and CDC
    surveillance publish counts, locations and clinical forms.

  * **Individual patient records: not public, and should not be.** Plague case
    counts are small enough that a real line list would be re-identifiable.
    Anyone offering one is either synthesising it or leaking it. PlagueShield
    therefore reconstructs case *vignettes* from published reports and labels
    them as such, rather than pretending to a line list it cannot have.

Every fetcher degrades gracefully: no network means a bundled snapshot or an
empty result with a recorded reason, never an exception that stops a clinical
assessment.
"""

from __future__ import annotations

import json
import pathlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

PUBLIC_DIR = pathlib.Path(__file__).resolve().parent / "public"

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "PlagueShield/0.1 (public-health decision support; contact: local)"


@dataclass
class FetchResult:
    source: str
    ok: bool
    fetched_at: str
    records: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "ok": self.ok,
            "fetched_at": self.fetched_at,
            "count": len(self.records),
            "records": self.records,
            "note": self.note,
            "url": self.url,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _client(timeout: float):
    import httpx

    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


def fetch_ncbi_biosamples(
    retmax: int = 60, timeout: float = 20.0
) -> FetchResult:
    """Public Y. pestis BioSample metadata via NCBI E-utilities.

    Used to characterise what the public isolate record actually looks like --
    geography, collection years, isolation sources. No credentials required.
    """
    query = "Yersinia pestis[Organism]"
    search_url = f"{NCBI_EUTILS}/esearch.fcgi"
    result = FetchResult(
        source="NCBI BioSample (Yersinia pestis)",
        ok=False,
        fetched_at=_now(),
        url="https://www.ncbi.nlm.nih.gov/biosample/?term=Yersinia+pestis",
    )

    try:
        with _client(timeout) as client:
            search = client.get(
                search_url,
                params={
                    "db": "biosample",
                    "term": query,
                    "retmax": str(retmax),
                    "retmode": "json",
                },
            )
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                result.note = "No identifiers returned"
                return result

            summary = client.get(
                f"{NCBI_EUTILS}/esummary.fcgi",
                params={
                    "db": "biosample",
                    "id": ",".join(ids[:retmax]),
                    "retmode": "json",
                },
            )
            summary.raise_for_status()
            payload = summary.json().get("result", {})

            for uid in payload.get("uids", []):
                entry = payload.get(uid, {})
                result.records.append(
                    {
                        "uid": uid,
                        "accession": entry.get("accession"),
                        "title": entry.get("title"),
                        "organism": entry.get("organism"),
                        "publication_date": entry.get("publicationdate"),
                        "sourcesample": entry.get("sourcesample"),
                    }
                )
        result.ok = True
        result.note = f"Retrieved {len(result.records)} public BioSample records"
    except Exception as exc:  # noqa: BLE001
        result.note = f"{type(exc).__name__}: {exc}"
    return result


def fetch_ncbi_assembly_counts(timeout: float = 20.0) -> FetchResult:
    """Counts public Y. pestis assemblies — calibrates the resistance prior.

    `plagueshield.fewshot.prior.CHARACTERISED_ISOLATES` is an
    order-of-magnitude figure; this is how it gets refreshed against reality.
    """
    result = FetchResult(
        source="NCBI Assembly (Yersinia pestis)",
        ok=False,
        fetched_at=_now(),
        url="https://www.ncbi.nlm.nih.gov/datasets/genome/?taxon=632",
    )
    try:
        with _client(timeout) as client:
            resp = client.get(
                f"{NCBI_EUTILS}/esearch.fcgi",
                params={
                    "db": "assembly",
                    "term": "Yersinia pestis[Organism]",
                    "retmax": "0",
                    "retmode": "json",
                },
            )
            resp.raise_for_status()
            count = int(resp.json()["esearchresult"]["count"])
        result.records = [{"public_assembly_count": count}]
        result.ok = True
        result.note = (
            f"{count} public Y. pestis assemblies — compare against "
            f"CHARACTERISED_ISOLATES in fewshot/prior.py"
        )
    except Exception as exc:  # noqa: BLE001
        result.note = f"{type(exc).__name__}: {exc}"
    return result


#: WHO serves Disease Outbreak News through an OData endpoint. The older
#: csr/don RSS feed is retired and returns 404.
WHO_DON_API = "https://www.who.int/api/news/diseaseoutbreaknews"
WHO_DON_BASE = "https://www.who.int/emergencies/disease-outbreak-news/item"


#: The API caps $top at 100, so a deep scan needs paging via $skip.
WHO_PAGE_SIZE = 100


def fetch_who_outbreak_news(
    timeout: float = 25.0, limit: int = 40, max_pages: int = 8
) -> FetchResult:
    """Plague items from WHO Disease Outbreak News.

    Plague DONs are infrequent — often only a handful across several years —
    so a single page of recent items would usually contain none at all. We
    page back through the feed by date and filter locally, keeping every
    plague item found plus a few recent items for context.
    """
    result = FetchResult(
        source="WHO Disease Outbreak News",
        ok=False,
        fetched_at=_now(),
        url="https://www.who.int/emergencies/disease-outbreak-news",
    )

    plague: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []
    scanned = 0

    try:
        with _client(timeout) as client:
            for page in range(max_pages):
                # Pre-encoded: the $ and the space in $orderby must survive.
                url = (
                    f"{WHO_DON_API}?%24orderby=PublicationDateAndTime%20desc"
                    f"&%24top={WHO_PAGE_SIZE}&%24skip={page * WHO_PAGE_SIZE}"
                    f"&%24format=json"
                )
                resp = client.get(url)
                resp.raise_for_status()
                items = resp.json().get("value", [])
                if not items:
                    break
                scanned += len(items)

                for item in items:
                    title = (item.get("Title") or "").strip()
                    slug = (item.get("ItemDefaultUrl") or "").strip("/")
                    entry = {
                        "title": title,
                        "url": f"{WHO_DON_BASE}/{slug}" if slug else None,
                        "published": item.get("PublicationDateAndTime"),
                        "summary": (item.get("Summary") or "").strip()[:600] or None,
                        "don_id": item.get("DonId"),
                    }
                    if "plague" in title.lower():
                        entry["plague_related"] = True
                        plague.append(entry)
                    elif len(recent) < 10:
                        entry["plague_related"] = False
                        recent.append(entry)

                if len(items) < WHO_PAGE_SIZE:
                    break

        result.records = (plague + recent)[:limit]
        result.ok = True
        result.note = (
            f"{scanned} DON items scanned across {max_pages} pages; "
            f"{len(plague)} plague-related found"
        )
    except Exception as exc:  # noqa: BLE001
        result.note = f"{type(exc).__name__}: {exc}"
        if plague or recent:
            result.records = (plague + recent)[:limit]
            result.note += f" (partial: {len(plague)} plague items retained)"
    return result


def fetch_all(timeout: float = 20.0) -> dict[str, FetchResult]:
    return {
        "ncbi_biosamples": fetch_ncbi_biosamples(timeout=timeout),
        "ncbi_assembly_counts": fetch_ncbi_assembly_counts(timeout=timeout),
        "who_outbreak_news": fetch_who_outbreak_news(timeout=timeout),
    }


def save_snapshots(results: dict[str, FetchResult]) -> list[pathlib.Path]:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for name, result in results.items():
        path = PUBLIC_DIR / f"{name}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2))
        written.append(path)
    return written
