from fastapi import APIRouter, status

from app.api.schemas.health import HealthResponse

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="agentic-rag-api",
        version="1.0.0",
    )
