"""FastAPI application for interacting with Elasticsearch."""
import os
from typing import Any, Optional

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


class AskRequest(BaseModel):
    """Model for the /ask endpoint request body."""

    query: str


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok"}


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, Any]:
    """Translate a natural language request into an Elasticsearch query and execute it."""

    if ES_CLIENT is None:
        return {"error": "Elasticsearch host is not configured."}

    user_query = request.query or ""
    lowered_query = user_query.lower()

    filters: list[dict[str, Any]] = []
    explanations: list[str] = []

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
        explanations.append("Restricted results to the previous day because the query mentioned yesterday.")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
