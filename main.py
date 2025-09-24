"""FastAPI application for interacting with Elasticsearch."""
import base64
import io
import os
from copy import deepcopy
from typing import Any, Optional

import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

ES_HOST_ENV = "ES_HOST"
ES_USER_ENV = "ES_USER"
ES_PASS_ENV = "ES_PASS"


def _load_env_variable(name: str) -> Optional[str]:
    """Load an environment variable.

    Parameters
    ----------
    name: str
        Name of the environment variable to load.

    Returns
    -------
    Optional[str]
        The value of the environment variable if set, otherwise ``None``.
    """

    return os.environ.get(name)


ES_HOST = _load_env_variable(ES_HOST_ENV)
ES_USER = _load_env_variable(ES_USER_ENV)
ES_PASS = _load_env_variable(ES_PASS_ENV)

# Initialize Elasticsearch client using the loaded credentials.
# The client is ready for future enhancements that may require Elasticsearch operations.
ES_CLIENT = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASS) if ES_USER or ES_PASS else None,
) if ES_HOST else None

app = FastAPI()

# Stores per-session context such as the last executed DSL for that session.
SESSION_CONTEXT: dict[str, dict[str, Any]] = {}

TimeframeRule = tuple[tuple[str, ...], dict[str, Any], str, str]

_TIMEFRAME_RULES: tuple[TimeframeRule, ...] = (
    (
        ("last month",),
        {
            "range": {
                "@timestamp": {
                    "gte": "now-30d/d",
                    "lt": "now/d",
                }
            }
        },
        "the last 30 days",
        "Restricted results to the last 30 days because the query mentioned last month.",
    ),
    (
        ("last week",),
        {
            "range": {
                "@timestamp": {
                    "gte": "now-7d/d",
                    "lt": "now/d",
                }
            }
        },
        "the last 7 days",
        "Restricted results to the last 7 days because the query mentioned last week.",
    ),
    (
        ("yesterday",),
        {
            "range": {
                "@timestamp": {
                    "gte": "now-1d/d",
                    "lt": "now/d",
                }
            }
        },
        "the previous day",
        "Restricted results to the previous day because the query mentioned yesterday.",
    ),
)


def _resolve_timeframe_filters(
    lowered_text: str,
) -> tuple[list[dict[str, Any]], str, Optional[str]]:
    """Return filters, description, and explanation derived from timeframe keywords."""

    for keywords, filter_template, description, explanation in _TIMEFRAME_RULES:
        if any(keyword in lowered_text for keyword in keywords):
            return [deepcopy(filter_template)], description, explanation

    return [], "the selected period", None


class AskRequest(BaseModel):
    """Model for the /ask endpoint request body."""

    session_id: str
    query: str


class ReportRequest(BaseModel):
    """Model for the /report endpoint request body."""

    session_id: str
    instruction: str
    index: str = "logs-*"


def _build_report_search_params(instruction: str, index: str) -> tuple[dict[str, Any], str, str]:
    """Translate a reporting instruction into Elasticsearch search parameters.

    Returns the search parameters, the subject being counted, and a human-readable
    timeframe description.
    """

    lowered = instruction.lower() if instruction else ""
    filters: list[dict[str, Any]] = []

    subject = "events"
    if "failed login" in lowered:
        filters.append({"term": {"event.action.keyword": "authentication_failure"}})
        subject = "failed logins"

    timeframe_filters, timeframe_description, _ = _resolve_timeframe_filters(lowered)
    filters.extend(timeframe_filters)

    if filters:
        query: dict[str, Any] = {"bool": {"filter": filters}}
    else:
        query = {"match_all": {}}

    search_params: dict[str, Any] = {
        "index": index or "logs-*",
        "size": 0,
        "query": query,
        "aggs": {
            "per_day": {
                "date_histogram": {
                    "field": "@timestamp",
                    "calendar_interval": "day",
                    "format": "yyyy-MM-dd",
                }
            }
        },
    }

    return search_params, subject, timeframe_description


def _generate_chart_base64(buckets: list[dict[str, Any]]) -> str:
    """Create a bar chart from aggregation buckets and return it as a base64 string."""

    dates = [bucket.get("key_as_string", str(bucket.get("key", ""))) for bucket in buckets]
    dates = [date.split("T")[0] if isinstance(date, str) else str(date) for date in dates]
    counts = [bucket.get("doc_count", 0) for bucket in buckets]

    fig, ax = plt.subplots(figsize=(6, 3))

    if counts:
        ax.bar(dates, counts, color="#1f77b4")
        ax.set_ylabel("Count")
        ax.set_xlabel("Date")
        ax.set_title("Daily counts")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    """Translate a natural language request into an Elasticsearch query and execute it."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    session_id = request.session_id
    session_data = SESSION_CONTEXT.setdefault(session_id, {})

    user_query = request.query or ""
    lowered_query = user_query.lower()

    filters: list[dict[str, Any]] = []
    explanations: list[str] = []

    apply_vpn_filter = "now filter only vpn" in lowered_query

    if apply_vpn_filter and session_data.get("last_dsl"):
        search_params = deepcopy(session_data["last_dsl"])
        query_clause = search_params.setdefault("query", {})
        if isinstance(query_clause, dict) and "bool" in query_clause:
            bool_query = query_clause["bool"]
            must_clause = bool_query.get("must")
            vpn_match = {"match": {"url.original": "vpn"}}
            if isinstance(must_clause, list):
                bool_query["must"] = [*must_clause, vpn_match]
            elif must_clause is None:
                bool_query["must"] = [vpn_match]
            else:
                bool_query["must"] = [must_clause, vpn_match]
        else:
            search_params["query"] = {"bool": {"must": [{"match": {"url.original": "vpn"}}]}}
        explanations.append("Applied VPN filter on previous query.")
    else:
        if apply_vpn_filter and not session_data.get("last_dsl"):
            explanations.append(
                "No previous query found for this session; unable to apply VPN filter."
            )

        if "failed login" in lowered_query or "suspicious login" in lowered_query:
            filters.append({"term": {"event.action.keyword": "authentication_failure"}})
            explanations.append(
                "Applied authentication failure filter because the query mentioned failed or suspicious logins."
            )

        timeframe_filters, _, timeframe_explanation = _resolve_timeframe_filters(lowered_query)
        if timeframe_filters:
            filters.extend(timeframe_filters)
            if timeframe_explanation:
                explanations.append(timeframe_explanation)

        if filters:
            es_query: dict[str, Any] = {"bool": {"filter": filters}}
        else:
            es_query = {"match_all": {}}
            explanations.append("No specific rule matched; returning the most recent documents.")

        search_params = {
            "index": "logs-*",
            "size": 5,
            "sort": [{"@timestamp": {"order": "desc"}}],
            "query": es_query,
        }

    SESSION_CONTEXT[session_id]["last_dsl"] = deepcopy(search_params)

    try:
        response = await run_in_threadpool(ES_CLIENT.search, **search_params)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}

    hits = response.get("hits", {}).get("hits", [])
    sources = [hit.get("_source", {}) for hit in hits]

    return {
        "dsl": search_params,
        "results": sources,
        "explain": " ".join(explanations),
    }


@app.post("/report")
async def report(request: ReportRequest) -> dict[str, Any]:
    """Generate an aggregation report based on a natural language instruction."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    session_data = SESSION_CONTEXT.setdefault(request.session_id, {})

    search_params, subject, timeframe_description = _build_report_search_params(
        request.instruction, request.index
    )
    session_data["last_report_dsl"] = deepcopy(search_params)

    try:
        response = await run_in_threadpool(ES_CLIENT.search, **search_params)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}

    aggregations = response.get("aggregations", {})
    per_day = aggregations.get("per_day", {})
    buckets = per_day.get("buckets", [])

    total_count = sum(bucket.get("doc_count", 0) for bucket in buckets)
    if buckets:
        peak_bucket = max(buckets, key=lambda bucket: bucket.get("doc_count", 0))
        peak_count = peak_bucket.get("doc_count", 0)
        peak_date_raw = peak_bucket.get("key_as_string") or str(peak_bucket.get("key", "N/A"))
        peak_date = peak_date_raw.split("T")[0] if isinstance(peak_date_raw, str) else str(peak_date_raw)
    else:
        peak_count = 0
        peak_date = "N/A"

    summary = (
        f"Found {total_count} {subject} in {timeframe_description}. "
        f"Peak was {peak_count} on date {peak_date}."
    )

    chart_b64 = _generate_chart_base64(buckets)

    return {
        "dsl": search_params,
        "summary": summary,
        "chart": chart_b64,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
