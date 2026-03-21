from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from app.core.engine import engine
from app.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.utils.logger import logger

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if request.stream:
            return StreamingResponse(
                engine.generate_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return engine.generate(request)

    except ValueError as ve:
        logger.warning(f"Bad request: {ve}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": {
                    "message": str(ve),
                    "type": "invalid_request_error",
                    "param": "response_format",
                    "code": "invalid_schema",
                }
            },
        )
    except Exception as e:
        logger.error(f"Inference failure: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": f"Server generation error: {str(e)}",
                    "type": "server_error",
                    "param": None,
                    "code": "internal_error",
                }
            },
        )
