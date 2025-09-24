"""FastAPI application for interacting with Elasticsearch."""
import base64
import io
import json
import os
import tempfile
from copy import deepcopy
from typing import Any, Optional
from xml.sax.saxutils import escape

import google.generativeai as genai
import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from starlette.background import BackgroundTask

ES_HOST_ENV = "ES_HOST"
ES_USER_ENV = "ES_USER"
ES_PASS_ENV = "ES_PASS"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "GEMINI_MODEL"


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
GEMINI_API_KEY = _load_env_variable(GEMINI_API_KEY_ENV)
GEMINI_MODEL = _load_env_variable(GEMINI_MODEL_ENV) or "gemini-1.5-flash"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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


async def _load_index_mapping(index: str) -> dict[str, Any]:
    """Fetch the index mapping for validation and prompting."""

    if ES_CLIENT is None:
        return {}

    try:
        return await run_in_threadpool(ES_CLIENT.indices.get_mapping, index=index)
    except Exception:  # noqa: BLE001 - validation should not break the flow
        return {}


def _format_mapping_snippet(mapping: dict[str, Any], max_chars: int = 2500) -> str:
    """Return a truncated JSON representation of the mapping for prompts."""

    if not mapping:
        return "Mapping unavailable."

    snippet_source: dict[str, Any] = {}
    for index_name, index_data in mapping.items():
        properties = index_data.get("mappings", {}).get("properties") or {}
        snippet_source[index_name] = properties
        break

    snippet = json.dumps(snippet_source, indent=2, sort_keys=True)
    if len(snippet) > max_chars:
        return f"{snippet[: max_chars - 3]}..."
    return snippet


def _collect_mapping_fields(mapping: dict[str, Any]) -> set[str]:
    """Flatten Elasticsearch mapping into a set of dot-notated field names."""

    fields: set[str] = set()

    def _walk(properties: dict[str, Any], prefix: str = "") -> None:
        for field_name, definition in properties.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            fields.add(full_name)

            if isinstance(definition, dict):
                sub_fields = definition.get("fields", {})
                if isinstance(sub_fields, dict):
                    for sub_name in sub_fields:
                        fields.add(f"{full_name}.{sub_name}")

                nested_properties = definition.get("properties")
                if isinstance(nested_properties, dict):
                    _walk(nested_properties, full_name)

    for index_data in mapping.values():
        properties = index_data.get("mappings", {}).get("properties")
        if isinstance(properties, dict):
            _walk(properties)

    return fields


def _extract_fields_from_dsl(obj: Any) -> set[str]:
    """Recursively collect field names referenced in a DSL structure."""

    fields: set[str] = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"term", "match", "match_phrase", "match_phrase_prefix", "wildcard", "regexp", "prefix"}:
                if isinstance(value, dict):
                    for field_name, field_value in value.items():
                        fields.add(field_name)
                        fields.update(_extract_fields_from_dsl(field_value))
                continue

            if key in {"range", "fuzzy"}:
                if isinstance(value, dict):
                    for field_name, field_value in value.items():
                        fields.add(field_name)
                        fields.update(_extract_fields_from_dsl(field_value))
                continue

            if key == "exists":
                if isinstance(value, dict):
                    field_name = value.get("field")
                    if isinstance(field_name, str):
                        fields.add(field_name)
                continue

            if key == "multi_match" and isinstance(value, dict):
                possible_fields = value.get("fields")
                if isinstance(possible_fields, list):
                    for candidate in possible_fields:
                        if isinstance(candidate, str):
                            fields.add(candidate.split("^")[0])
                default_field = value.get("default_field")
                if isinstance(default_field, str):
                    fields.add(default_field)
                fields.update(_extract_fields_from_dsl(value))
                continue

            if key == "query_string" and isinstance(value, dict):
                qs_fields = value.get("fields")
                if isinstance(qs_fields, list):
                    for candidate in qs_fields:
                        if isinstance(candidate, str):
                            fields.add(candidate.split("^")[0])
                default_field = value.get("default_field")
                if isinstance(default_field, str):
                    fields.add(default_field)
                fields.update(_extract_fields_from_dsl(value))
                continue

            if key == "sort":
                if isinstance(value, list):
                    for sort_entry in value:
                        if isinstance(sort_entry, dict):
                            for field_name, sort_value in sort_entry.items():
                                if isinstance(field_name, str) and field_name != "_score":
                                    fields.add(field_name)
                                fields.update(_extract_fields_from_dsl(sort_value))
                elif isinstance(value, dict):
                    for field_name, sort_value in value.items():
                        if isinstance(field_name, str) and field_name != "_score":
                            fields.add(field_name)
                        fields.update(_extract_fields_from_dsl(sort_value))
                continue

            if key in {"aggs", "aggregations"}:
                fields.update(_extract_fields_from_dsl(value))
                continue

            if key == "field" and isinstance(value, str):
                fields.add(value)
                continue

            if key in {"fields", "docvalue_fields"} and isinstance(value, list):
                for candidate in value:
                    if isinstance(candidate, str):
                        fields.add(candidate.split("^")[0])
                continue

            fields.update(_extract_fields_from_dsl(value))

    elif isinstance(obj, list):
        for item in obj:
            fields.update(_extract_fields_from_dsl(item))

    return fields


def _validate_dsl_fields(dsl: dict[str, Any], mapping_fields: set[str]) -> set[str]:
    """Return the set of fields referenced in DSL that are absent in the mapping."""

    if not mapping_fields:
        return set()

    referenced = _extract_fields_from_dsl(dsl)
    return {field for field in referenced if field not in mapping_fields}


def translate_with_gemini(
    nl_text: str,
    mapping: dict[str, Any],
    prefer: str = "dsl",
) -> tuple[dict[str, Any], str]:
    """Translate natural language into Elasticsearch DSL using Gemini."""

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    mapping_snippet = _format_mapping_snippet(mapping)
    example_translation = {
        "index": "logs-*",
        "size": 5,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"event.action.keyword": "authentication_failure"}},
                    {
                        "range": {
                            "@timestamp": {
                                "gte": "now-1d/d",
                                "lt": "now/d",
                            }
                        }
                    },
                ]
            }
        },
    }

    prompt = (
        "You are an assistant that converts natural language security analytics requests "
        "into Elasticsearch DSL queries. Act as a query generator.\n\n"
        "Example translation:\n"
        "NL: \"Show failed login attempts in the last 24 hours\"\n"
        f"DSL:\n{json.dumps(example_translation, indent=2)}\n\n"
        "Index mapping snippet for logs-*:\n"
        f"{mapping_snippet}\n\n"
        "Translate the following natural language request into Elasticsearch DSL. "
        f"Prefer producing the {prefer}. Respond with JSON only using the keys "
        '{"dsl": { ... }, "explain": "<text>"}. Do not wrap the response in markdown.'
        "\nNL input: "
        f"{nl_text}"
    )

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )

    response = model.generate_content(prompt)
    if not getattr(response, "text", None):
        raise ValueError("Empty response from Gemini.")

    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:  # noqa: PERF203 - more informative error
        raise ValueError("Gemini response was not valid JSON.") from exc

    dsl = payload.get("dsl")
    if not isinstance(dsl, dict):
        raise ValueError("Gemini response missing 'dsl' object.")

    explain_text = payload.get("explain", "")
    if explain_text is None:
        explain_text = ""

    return dsl, str(explain_text)


def _rule_based_translation(
    session_data: dict[str, Any],
    user_query: str,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Attempt to translate using local rule-based heuristics."""

    lowered_query = (user_query or "").lower()
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
        return search_params, explanations

    if apply_vpn_filter and not session_data.get("last_dsl"):
        explanations.append(
            "No previous query found for this session; unable to apply VPN filter."
        )

    filters: list[dict[str, Any]] = []
    matched_rule = False

    if "failed login" in lowered_query or "suspicious login" in lowered_query:
        filters.append({"term": {"event.action.keyword": "authentication_failure"}})
        explanations.append(
            "Applied authentication failure filter because the query mentioned failed or suspicious logins."
        )
        matched_rule = True

    timeframe_filters, _, timeframe_explanation = _resolve_timeframe_filters(lowered_query)
    if timeframe_filters:
        filters.extend(timeframe_filters)
        matched_rule = True
        if timeframe_explanation:
            explanations.append(timeframe_explanation)

    if matched_rule:
        if filters:
            es_query: dict[str, Any] = {"bool": {"filter": filters}}
        else:
            es_query = {"match_all": {}}

        search_params = {
            "index": "logs-*",
            "size": 5,
            "sort": [{"@timestamp": {"order": "desc"}}],
            "query": es_query,
        }
        return search_params, explanations

    return None, explanations


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


def _create_chart_image_bytes(buckets: list[dict[str, Any]]) -> bytes:
    """Create a bar chart from aggregation buckets and return it as PNG bytes."""

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
    return buffer.read()


async def _generate_report_payload(request: ReportRequest) -> dict[str, Any]:
    """Execute the report search and construct the reusable response payload."""

    session_data = SESSION_CONTEXT.setdefault(request.session_id, {})

    search_params, subject, timeframe_description = _build_report_search_params(
        request.instruction, request.index
    )
    session_data["last_report_dsl"] = deepcopy(search_params)

    response = await run_in_threadpool(ES_CLIENT.search, **search_params)

    aggregations = response.get("aggregations", {})
    per_day = aggregations.get("per_day", {})
    buckets = per_day.get("buckets", [])

    total_count = sum(bucket.get("doc_count", 0) for bucket in buckets)
    if buckets:
        peak_bucket = max(buckets, key=lambda bucket: bucket.get("doc_count", 0))
        peak_count = peak_bucket.get("doc_count", 0)
        peak_date_raw = peak_bucket.get("key_as_string") or str(peak_bucket.get("key", "N/A"))
        peak_date = (
            peak_date_raw.split("T")[0]
            if isinstance(peak_date_raw, str)
            else str(peak_date_raw)
        )
    else:
        peak_count = 0
        peak_date = "N/A"

    summary = (
        f"Found {total_count} {subject} in {timeframe_description}. "
        f"Peak was {peak_count} on date {peak_date}."
    )

    chart_bytes = _create_chart_image_bytes(buckets)

    return {
        "dsl": search_params,
        "summary": summary,
        "buckets": buckets,
        "chart_bytes": chart_bytes,
    }


def _build_report_filename(instruction: str) -> str:
    """Create a filesystem-friendly filename for the generated report."""

    cleaned = "".join(
        char if char.isalnum() else "_" for char in (instruction or "").strip()
    )
    cleaned = cleaned or "report"
    return f"report_{cleaned}.pdf"


def _build_pdf_report(
    instruction: str, summary: str, buckets: list[dict[str, Any]], chart_bytes: bytes
) -> str:
    """Create a PDF report and return the filesystem path to the generated file."""

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.close()

    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()

    elements = []

    title_text = f"Report for {instruction}" if instruction else "Report"
    elements.append(Paragraph(escape(title_text), styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(escape(summary), styles["BodyText"]))
    elements.append(Spacer(1, 12))

    table_data: list[list[str]] = [["Date", "Count"]]
    if buckets:
        for bucket in buckets:
            raw_date = bucket.get("key_as_string") or str(bucket.get("key", ""))
            date = raw_date.split("T")[0] if isinstance(raw_date, str) else str(raw_date)
            table_data.append([date, str(bucket.get("doc_count", 0))])
    else:
        table_data.append(["No data", "0"])

    table = Table(table_data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 12))

    chart_stream = io.BytesIO(chart_bytes)
    chart_stream.seek(0)
    elements.append(RLImage(chart_stream, width=6 * inch, height=3 * inch))

    doc.build(elements)

    return temp_file.name


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

    mapping = await _load_index_mapping("logs-*")

    search_params, explanations = _rule_based_translation(session_data, user_query)

    if search_params is None:
        if not GEMINI_API_KEY:
            base_explain = " ".join(explanations).strip()
            response: dict[str, Any] = {
                "error": "No rule matched and GEMINI_API_KEY is not configured.",
            }
            if base_explain:
                response["explain"] = base_explain
            return response

        try:
            translated_dsl, llm_explain = translate_with_gemini(user_query, mapping)
        except Exception as exc:  # noqa: BLE001 - we want to surface translation errors
            base_explain = " ".join(explanations).strip()
            error_payload: dict[str, Any] = {
                "error": f"Gemini translation failed: {exc}",
            }
            if base_explain:
                error_payload["explain"] = base_explain
            return error_payload

        search_params = translated_dsl
        if llm_explain:
            explanations.append(llm_explain)

    if not isinstance(search_params, dict):
        return {"error": "Translated DSL must be an object."}

    if "index" not in search_params:
        search_params["index"] = "logs-*"

    mapping_fields = _collect_mapping_fields(mapping)
    missing_fields = _validate_dsl_fields(search_params, mapping_fields)
    if missing_fields:
        explain_text = " ".join(explanations).strip()
        return {
            "error": "Fields not found in mapping: "
            + ", ".join(sorted(missing_fields)),
            "dsl": search_params,
            "explain": explain_text,
        }

    SESSION_CONTEXT[session_id]["last_dsl"] = deepcopy(search_params)

    try:
        response = await run_in_threadpool(ES_CLIENT.search, **search_params)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}

    hits = response.get("hits", {}).get("hits", [])
    sources = [hit.get("_source", {}) for hit in hits]

    explain_text = " ".join(explanations).strip()

    return {
        "dsl": search_params,
        "results": sources,
        "explain": explain_text,
    }


@app.post("/report")
async def report(request: ReportRequest) -> dict[str, Any]:
    """Generate an aggregation report based on a natural language instruction."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    try:
        payload = await _generate_report_payload(request)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}

    chart_b64 = base64.b64encode(payload["chart_bytes"]).decode("utf-8")

    return {
        "dsl": payload["dsl"],
        "summary": payload["summary"],
        "chart": chart_b64,
    }


@app.post("/report/pdf")
async def report_pdf(request: ReportRequest) -> FileResponse | dict[str, str]:
    """Generate a PDF report using the same aggregation logic as ``/report``."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    try:
        payload = await _generate_report_payload(request)
    except Exception as exc:  # noqa: BLE001 - intentionally broad to surface error message
        return {"error": str(exc)}

    pdf_path = _build_pdf_report(
        request.instruction,
        payload["summary"],
        payload["buckets"],
        payload["chart_bytes"],
    )

    filename = _build_report_filename(request.instruction)
    background_task = BackgroundTask(os.remove, pdf_path)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        background=background_task,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
