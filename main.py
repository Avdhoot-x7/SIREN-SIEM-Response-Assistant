"""FastAPI application for interacting with Elasticsearch."""
import os
from typing import Optional

from elasticsearch import Elasticsearch
from fastapi import FastAPI
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
async def ask(request: AskRequest) -> dict[str, str]:
    """Accept a query string and echo back a placeholder response."""

    return {"message": "received", "query": request.query}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
