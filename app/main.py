import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.router import api_router


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Starting Agentic RAG API")

    yield

    logger.info("Shutting down Agentic RAG API")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic RAG API",
        description="Production API for the Agentic RAG platform",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(api_router)

    return app


app = create_app()