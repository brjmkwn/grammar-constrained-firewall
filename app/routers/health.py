from fastapi import APIRouter
from app.config import settings
from app.core.cache import fsm_cache
from app.schemas import HealthResponse, MetricsResponse
from app.utils.metrics import metrics

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    summary = metrics.get_summary()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_id=settings.MODEL_ID,
        cache_entries=fsm_cache.size,
        uptime_seconds=summary["uptime_seconds"],
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    summary = metrics.get_summary()
    return MetricsResponse(
        total_requests=summary["total_requests"],
        structured_requests=summary["structured_requests"],
        passthrough_requests=summary["passthrough_requests"],
        schema_validity_rate_pct=summary["schema_validity_rate_pct"],
        total_schema_violations_blocked=summary["total_schema_violations_blocked"],
        avg_masking_latency_ms=summary["avg_masking_latency_ms"],
        p95_masking_latency_ms=summary["p95_masking_latency_ms"],
        p99_masking_latency_ms=summary["p99_masking_latency_ms"],
        cache_hit_rate_pct=summary["cache_hit_rate_pct"],
    )
