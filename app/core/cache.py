import hashlib
import json
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union
from interegular.fsm import FSM
from pydantic import BaseModel

from app.core.compiler import compiler
from app.utils.logger import logger
from app.utils.metrics import metrics


class FSMCache:
    def __init__(self, capacity: int = 256):
        self.capacity = capacity
        self._cache: OrderedDict[str, FSM] = OrderedDict()
        self._lock = threading.Lock()

    def _compute_key(self, schema_or_regex: Union[Dict[str, Any], str, type]) -> str:
        if isinstance(schema_or_regex, str):
            payload = schema_or_regex
        elif isinstance(schema_or_regex, type) and issubclass(schema_or_regex, BaseModel):
            payload = json.dumps(schema_or_regex.model_json_schema(), sort_keys=True)
        elif isinstance(schema_or_regex, dict):
            payload = json.dumps(schema_or_regex, sort_keys=True)
        else:
            payload = str(schema_or_regex)

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_or_compile(
        self,
        schema_or_regex: Union[Dict[str, Any], str, type],
        is_regex: bool = False,
    ) -> FSM:
        key = self._compute_key(schema_or_regex)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                metrics.record_cache_lookup(hit=True)
                return self._cache[key]

        if is_regex and isinstance(schema_or_regex, str):
            fsm = compiler.compile_regex_to_fsm(schema_or_regex)
        else:
            fsm = compiler.compile_schema_to_fsm(schema_or_regex)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                metrics.record_cache_lookup(hit=True)
                return self._cache[key]

            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)

            self._cache[key] = fsm
            metrics.record_cache_lookup(hit=False)
            return fsm

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


fsm_cache = FSMCache()
