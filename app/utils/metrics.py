import threading
import time
from collections import deque
from typing import Deque, Dict, List
import numpy as np


class MetricsTracker:
    def __init__(self, max_history: int = 10000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_requests = 0
        self.structured_requests = 0
        self.passthrough_requests = 0
        self.valid_schema_generations = 0
        self.invalid_schema_generations = 0
        self.total_violations_blocked = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.masking_latencies_ms: Deque[float] = deque(maxlen=max_history)

    def record_request(self, is_structured: bool):
        with self._lock:
            self.total_requests += 1
            if is_structured:
                self.structured_requests += 1
            else:
                self.passthrough_requests += 1

    def record_schema_result(self, is_valid: bool, violations_blocked: int = 0):
        with self._lock:
            if is_valid:
                self.valid_schema_generations += 1
            else:
                self.invalid_schema_generations += 1
            self.total_violations_blocked += violations_blocked

    def record_masking_latency(self, latency_ms: float):
        with self._lock:
            self.masking_latencies_ms.append(latency_ms)

    def record_cache_lookup(self, hit: bool):
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def get_summary(self) -> Dict[str, float]:
        with self._lock:
            total_cache = self.cache_hits + self.cache_misses
            cache_hit_rate = (self.cache_hits / total_cache * 100.0) if total_cache > 0 else 0.0

            total_structured = self.valid_schema_generations + self.invalid_schema_generations
            validity_rate = (self.valid_schema_generations / total_structured * 100.0) if total_structured > 0 else 100.0

            latencies = list(self.masking_latencies_ms)
            if latencies:
                avg_lat = float(np.mean(latencies))
                p95_lat = float(np.percentile(latencies, 95))
                p99_lat = float(np.percentile(latencies, 99))
            else:
                avg_lat = 0.0
                p95_lat = 0.0
                p99_lat = 0.0

            return {
                "total_requests": self.total_requests,
                "structured_requests": self.structured_requests,
                "passthrough_requests": self.passthrough_requests,
                "schema_validity_rate_pct": validity_rate,
                "total_schema_violations_blocked": self.total_violations_blocked,
                "avg_masking_latency_ms": round(avg_lat, 4),
                "p95_masking_latency_ms": round(p95_lat, 4),
                "p99_masking_latency_ms": round(p99_lat, 4),
                "cache_hit_rate_pct": round(cache_hit_rate, 2),
                "uptime_seconds": round(time.time() - self.start_time, 2),
            }


metrics = MetricsTracker()
