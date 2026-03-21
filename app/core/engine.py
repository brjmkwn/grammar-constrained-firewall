import json
import random
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
import numpy as np

from app.config import settings
from app.core.cache import fsm_cache
from app.core.compiler import compiler
from app.core.masker import TokenLogitMasker, default_vocab, VocabularyMapper, NEG_INF
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    ChunkDelta,
    UsageInfo,
)
from app.utils.logger import logger
from app.utils.metrics import metrics

_BASE_LOGITS_TEMPLATE = np.random.normal(loc=0.0, scale=1.0, size=min(default_vocab.vocab_size, 100256))
for char in ['{', '}', '[', ']', ':', ',', '"', ' ', '0', '1', '2', '3', 'a', 'b', 'c', 't', 'f', 'A', 'S']:
    for tid in default_vocab.encode(char):
        if tid < len(_BASE_LOGITS_TEMPLATE):
            _BASE_LOGITS_TEMPLATE[tid] += 4.0


class ConstrainedGenerationEngine:
    def __init__(self, vocab: Optional[VocabularyMapper] = None):
        self.vocab = vocab or default_vocab

    def _extract_schema(self, request: ChatCompletionRequest) -> Optional[Dict[str, Any]]:
        if not request.response_format:
            return None

        rf = request.response_format
        if isinstance(rf, dict):
            rf_type = rf.get("type")
            if rf_type == "json_schema":
                js = rf.get("json_schema", {})
                return js.get("schema")
            elif rf_type == "json_object":
                return {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }
        elif hasattr(rf, "type"):
            if rf.type == "json_schema" and rf.json_schema:
                return rf.json_schema.schema_
            elif rf.type == "json_object":
                return {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                }

        return None

    def _get_raw_logits(
        self,
        prompt_tokens: List[int],
        generated_tokens: List[int],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        return _BASE_LOGITS_TEMPLATE

    def _sample_unconstrained_token(
        self,
        logits: np.ndarray,
        temperature: float = 0.7,
        top_p: float = 1.0,
    ) -> int:
        if temperature <= 1e-4:
            return int(np.argmax(logits))

        top_k = 50
        top_indices = np.argpartition(logits, -top_k)[-top_k:]
        sub_logits = logits[top_indices] / max(temperature, 1e-4)
        exp_logits = np.exp(sub_logits - np.max(sub_logits))
        probs = exp_logits / np.sum(exp_logits)

        choice = np.random.choice(len(top_indices), p=probs)
        return int(top_indices[choice])

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        t_start = time.time()
        schema_dict = self._extract_schema(request)
        is_structured = schema_dict is not None
        metrics.record_request(is_structured=is_structured)

        prompt_text = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
        prompt_tokens = self.vocab.encode(prompt_text)

        masker: Optional[TokenLogitMasker] = None
        if is_structured and schema_dict is not None:
            fsm = fsm_cache.get_or_compile(schema_dict)
            masker = TokenLogitMasker(fsm=fsm, vocab=self.vocab)

        generated_tokens: List[int] = []
        max_tokens = request.max_tokens if request.max_tokens is not None else (
            settings.DEFAULT_MAX_TOKENS if is_structured else 32
        )

        temperature = request.temperature if request.temperature is not None else settings.DEFAULT_TEMPERATURE
        violations_blocked = 0

        for step in range(max_tokens):
            raw_logits = self._get_raw_logits(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                seed=request.seed,
            )

            if masker is not None:
                unconstrained_top_token = int(np.argmax(raw_logits))

                chosen_token, latency_ms = masker.sample_valid_token(
                    raw_logits=raw_logits,
                    temperature=temperature,
                    top_p=request.top_p or 1.0,
                )
                metrics.record_masking_latency(latency_ms)

                if unconstrained_top_token != chosen_token:
                    violations_blocked += 1

                if chosen_token == self.vocab.eos_token_id:
                    break

                masker.step(chosen_token)
                generated_tokens.append(chosen_token)

                if masker.is_finished():
                    try:
                        text_so_far = self.vocab.decode(generated_tokens)
                        json.loads(text_so_far)
                        break
                    except json.JSONDecodeError:
                        pass
            else:
                chosen_token = self._sample_unconstrained_token(
                    logits=raw_logits,
                    temperature=temperature,
                    top_p=request.top_p or 1.0,
                )
                if chosen_token == self.vocab.eos_token_id:
                    break
                generated_tokens.append(chosen_token)

        output_text = self.vocab.decode(generated_tokens)

        if is_structured:
            try:
                json.loads(output_text)
                metrics.record_schema_result(is_valid=True, violations_blocked=violations_blocked)
            except Exception as e:
                logger.error(f"Schema violation: {e}", extra={"text": output_text})
                metrics.record_schema_result(is_valid=False, violations_blocked=violations_blocked)

        req_id = f"chatcmpl-{int(time.time() * 1000)}"
        return ChatCompletionResponse(
            id=req_id,
            created=int(t_start),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_tokens),
                completion_tokens=len(generated_tokens),
                total_tokens=len(prompt_tokens) + len(generated_tokens),
            ),
        )

    async def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncGenerator[str, None]:
        t_start = time.time()
        schema_dict = self._extract_schema(request)
        is_structured = schema_dict is not None
        metrics.record_request(is_structured=is_structured)

        prompt_text = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
        prompt_tokens = self.vocab.encode(prompt_text)

        masker: Optional[TokenLogitMasker] = None
        if is_structured and schema_dict is not None:
            fsm = fsm_cache.get_or_compile(schema_dict)
            masker = TokenLogitMasker(fsm=fsm, vocab=self.vocab)

        generated_tokens: List[int] = []
        max_tokens = request.max_tokens if request.max_tokens is not None else (
            settings.DEFAULT_MAX_TOKENS if is_structured else 32
        )

        temperature = request.temperature if request.temperature is not None else settings.DEFAULT_TEMPERATURE
        req_id = f"chatcmpl-{int(time.time() * 1000)}"

        initial_chunk = ChatCompletionChunk(
            id=req_id,
            created=int(t_start),
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChunkDelta(role="assistant", content=""), finish_reason=None)],
        )
        yield f"data: {initial_chunk.model_dump_json()}\n\n"

        for step in range(max_tokens):
            raw_logits = self._get_raw_logits(
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                seed=request.seed,
            )

            if masker is not None:
                chosen_token, latency_ms = masker.sample_valid_token(
                    raw_logits=raw_logits,
                    temperature=temperature,
                    top_p=request.top_p or 1.0,
                )
                metrics.record_masking_latency(latency_ms)

                if chosen_token == self.vocab.eos_token_id:
                    break

                masker.step(chosen_token)
                generated_tokens.append(chosen_token)
                token_text = self.vocab.decode([chosen_token])

                chunk = ChatCompletionChunk(
                    id=req_id,
                    created=int(t_start),
                    model=request.model,
                    choices=[ChunkChoice(index=0, delta=ChunkDelta(content=token_text), finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

                if masker.is_finished():
                    try:
                        text_so_far = self.vocab.decode(generated_tokens)
                        json.loads(text_so_far)
                        break
                    except json.JSONDecodeError:
                        pass
            else:
                chosen_token = self._sample_unconstrained_token(
                    logits=raw_logits,
                    temperature=temperature,
                    top_p=request.top_p or 1.0,
                )
                if chosen_token == self.vocab.eos_token_id:
                    break

                generated_tokens.append(chosen_token)
                token_text = self.vocab.decode([chosen_token])

                chunk = ChatCompletionChunk(
                    id=req_id,
                    created=int(t_start),
                    model=request.model,
                    choices=[ChunkChoice(index=0, delta=ChunkDelta(content=token_text), finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        final_chunk = ChatCompletionChunk(
            id=req_id,
            created=int(t_start),
            model=request.model,
            choices=[ChunkChoice(index=0, delta=ChunkDelta(), finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


engine = ConstrainedGenerationEngine()
