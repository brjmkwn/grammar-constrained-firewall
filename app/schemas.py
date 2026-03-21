from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function", "tool"]
    content: str
    name: Optional[str] = None


class JsonSchemaSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = None
    description: Optional[str] = None
    strict: Optional[bool] = True
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: Optional[JsonSchemaSpec] = None


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-2.5-7b-instruct"
    messages: List[ChatMessage]
    response_format: Optional[Union[ResponseFormat, Dict[str, Any]]] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 2048
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    user: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = "fp_fsm_01"


class ChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChunkChoice]
    system_fingerprint: Optional[str] = "fp_fsm_01"


class HealthResponse(BaseModel):
    status: str
    version: str
    model_id: str
    cache_entries: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    structured_requests: int
    passthrough_requests: int
    schema_validity_rate_pct: float
    total_schema_violations_blocked: int
    avg_masking_latency_ms: float
    p95_masking_latency_ms: float
    p99_masking_latency_ms: float
    cache_hit_rate_pct: float
