"""FastAPI application for interacting with Elasticsearch."""
import io
import os
import re
from copy import deepcopy
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import matplotlib
from elasticsearch import Elasticsearch
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as PlatypusImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from starlette.background import BackgroundTask

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


class AskRequest(BaseModel):
    """Model for the /ask endpoint request body."""

    session_id: str
    query: str


class ReportRequest(BaseModel):
    """Model for report generation endpoints."""

    session_id: str
    instruction: str
    index: str = "logs-*"


def _resolve_report_query(session_id: str) -> dict[str, Any]:
    """Return the Elasticsearch query to use for a report."""

    session_data = SESSION_CONTEXT.get(session_id, {})
    last_dsl = session_data.get("last_dsl")
    if isinstance(last_dsl, dict):
        query = deepcopy(last_dsl.get("query"))
        if isinstance(query, dict) and query:
            return query
    return {"match_all": {}}


def _build_report_search_params(index: str, query: dict[str, Any]) -> dict[str, Any]:
    """Construct the Elasticsearch search parameters for the report."""

    return {
        "index": index,
        "size": 0,
        "query": query,
        "aggs": {
            "events_over_time": {
                "date_histogram": {
                    "field": "@timestamp",
                    "calendar_interval": "day",
                    "min_doc_count": 0,
                }
            }
        },
    }


def _format_bucket_date(bucket: dict[str, Any]) -> str:
    """Format the bucket date as YYYY-MM-DD."""

    key_as_string = bucket.get("key_as_string")
    if key_as_string:
        return key_as_string.split("T", 1)[0]

    key = bucket.get("key")
    if key is None:
        return "unknown"

    return datetime.utcfromtimestamp(key / 1000).strftime("%Y-%m-%d")


def _summarize_counts(counts: list[dict[str, Any]], instruction: str) -> str:
    """Create a textual summary for the report."""

    total = sum(item["count"] for item in counts)
    if not counts:
        return (
            f"No events were found to satisfy the instruction \"{instruction}\". "
            "Try adjusting the filters or time range."
        )

    start_date = counts[0]["date"]
    end_date = counts[-1]["date"]
    peak = max(counts, key=lambda item: item["count"])

    if start_date == end_date:
        range_description = f"on {start_date}"
    else:
        range_description = f"from {start_date} to {end_date}"

    return (
        f'The instruction "{instruction}" matched {total} events {range_description}. '
        f'The busiest day was {peak["date"]} with {peak["count"]} events.'
    )


async def _generate_report_data(request: ReportRequest) -> dict[str, Any]:
    """Execute the report aggregation and return structured data."""

    query = _resolve_report_query(request.session_id)
    search_params = _build_report_search_params(request.index, query)

    response = await run_in_threadpool(ES_CLIENT.search, **search_params)

    buckets = (
        response.get("aggregations", {})
        .get("events_over_time", {})
        .get("buckets", [])
    )

    counts = [
        {
            "date": _format_bucket_date(bucket),
            "count": int(bucket.get("doc_count", 0)),
        }
        for bucket in buckets
    ]

    summary = _summarize_counts(counts, request.instruction)

    return {
        "instruction": request.instruction,
        "summary": summary,
        "counts": counts,
        "index": request.index,
    }


def _create_chart(counts: list[dict[str, Any]]) -> bytes:
    """Render the counts per day as a PNG chart and return its bytes."""

    fig, ax = plt.subplots(figsize=(8, 3))

    dates = [item["date"] for item in counts]
    values = [item["count"] for item in counts]

    if dates:
        ax.plot(dates, values, marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Event count")
        ax.set_title("Events per day")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        fig.autofmt_xdate(rotation=45)
    else:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def _build_pdf(report_data: dict[str, Any], chart_bytes: bytes) -> str:
    """Generate a PDF report and return the path to the temporary file."""

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        doc = SimpleDocTemplate(tmp_file.name, pagesize=letter)
        styles = getSampleStyleSheet()
        story: list[Any] = []

        story.append(Paragraph(f"Report for {report_data['instruction']}", styles["Heading1"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(report_data["summary"], styles["BodyText"]))
        story.append(Spacer(1, 0.2 * inch))

        table_data = [["Date", "Count"]]
        if report_data["counts"]:
            table_data.extend(
                [[item["date"], str(item["count"])] for item in report_data["counts"]]
            )
        else:
            table_data.append(["No data", "0"])

        table = Table(table_data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        chart_buffer = io.BytesIO(chart_bytes)
        story.append(PlatypusImage(chart_buffer, width=6 * inch, height=3 * inch))

        doc.build(story)

        return tmp_file.name


def _sanitize_filename(text: str) -> str:
    """Return a filesystem-friendly filename for the generated PDF."""

    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    if not safe:
        safe = "report"
    safe = safe[:60]
    return f"{safe}.pdf"


def _remove_file(path: str) -> None:
    """Remove a file, ignoring errors."""

    try:
        os.remove(path)
    except OSError:
        pass


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

        if "yesterday" in lowered_query:
            filters.append(
                {
                    "range": {
                        "@timestamp": {
                            "gte": "now-1d/d",
                            "lt": "now/d",
                        }
                    }
                }
            )
            explanations.append(
                "Restricted results to the previous day because the query mentioned yesterday."
            )

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
    """Generate a JSON report using Elasticsearch aggregations."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    try:
        return await _generate_report_data(request)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}


@app.post("/report/pdf")
async def report_pdf(request: ReportRequest) -> FileResponse | dict[str, str]:
    """Generate a PDF version of the report."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    try:
        report_data = await _generate_report_data(request)
        chart_bytes = _create_chart(report_data["counts"])
        pdf_path = _build_pdf(report_data, chart_bytes)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}

    filename = _sanitize_filename(request.instruction)
    background = BackgroundTask(_remove_file, pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        background=background,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
