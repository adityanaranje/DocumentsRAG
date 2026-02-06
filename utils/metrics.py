"""
Application metrics collection and tracking.
"""
import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
from config import config


class MetricsCollector:
    """Thread-safe metrics collector."""
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # Request metrics
        self.request_count = 0
        self.request_latencies = deque(maxlen=1000)  # Keep last 1000
        self.request_errors = 0
        
        # Intent distribution
        self.intent_counts = defaultdict(int)
        
        # LLM metrics
        self.llm_call_count = 0
        self.llm_cache_hits = 0
        self.llm_cache_misses = 0
        self.llm_latencies = deque(maxlen=1000)
        self.llm_errors = 0
        
        # Retrieval metrics
        self.retrieval_count = 0
        self.retrieval_latencies = deque(maxlen=1000)
        self.retrieval_empty_results = 0
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Circuit breaker metrics
        self.circuit_breaker_opens = 0
        self.circuit_breaker_failures = 0
        
        # Active requests
        self.active_requests = 0
        
        # Start time
        self.start_time = datetime.now()
    
    def record_request(self, latency_ms: float, intent: Optional[str] = None, error: bool = False):
        """Record a request."""
        with self._lock:
            self.request_count += 1
            self.request_latencies.append(latency_ms)
            
            if error:
                self.request_errors += 1
            
            if intent:
                self.intent_counts[intent] += 1
    
    def record_llm_call(self, latency_ms: float, cache_hit: bool = False, error: bool = False):
        """Record an LLM call."""
        with self._lock:
            self.llm_call_count += 1
            self.llm_latencies.append(latency_ms)
            
            if cache_hit:
                self.llm_cache_hits += 1
            else:
                self.llm_cache_misses += 1
            
            if error:
                self.llm_errors += 1
    
    def record_retrieval(self, latency_ms: float, result_count: int):
        """Record a retrieval operation."""
        with self._lock:
            self.retrieval_count += 1
            self.retrieval_latencies.append(latency_ms)
            
            if result_count == 0:
                self.retrieval_empty_results += 1
    
    def record_cache_access(self, hit: bool):
        """Record cache access."""
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def record_circuit_breaker_event(self, opened: bool = False, failure: bool = False):
        """Record circuit breaker event."""
        with self._lock:
            if opened:
                self.circuit_breaker_opens += 1
            if failure:
                self.circuit_breaker_failures += 1
    
    def increment_active_requests(self):
        """Increment active request count."""
        with self._lock:
            self.active_requests += 1
    
    def decrement_active_requests(self):
        """Decrement active request count."""
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)
    
    def get_metrics(self) -> Dict:
        """Get all metrics as a dictionary."""
        with self._lock:
            uptime = datetime.now() - self.start_time
            
            # Calculate percentiles for latencies
            req_latencies_sorted = sorted(self.request_latencies) if self.request_latencies else [0]
            llm_latencies_sorted = sorted(self.llm_latencies) if self.llm_latencies else [0]
            ret_latencies_sorted = sorted(self.retrieval_latencies) if self.retrieval_latencies else [0]
            
            def percentile(data, p):
                if not data:
                    return 0
                k = (len(data) - 1) * p
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return data[f] * (1 - c) + data[f + 1] * c
                return data[f]
            
            return {
                "uptime_seconds": uptime.total_seconds(),
                "timestamp": datetime.now().isoformat(),
                
                # Request metrics
                "requests": {
                    "total": self.request_count,
                    "active": self.active_requests,
                    "errors": self.request_errors,
                    "error_rate": self.request_errors / max(1, self.request_count),
                    "latency_ms": {
                        "min": min(req_latencies_sorted),
                        "max": max(req_latencies_sorted),
                        "p50": percentile(req_latencies_sorted, 0.50),
                        "p95": percentile(req_latencies_sorted, 0.95),
                        "p99": percentile(req_latencies_sorted, 0.99),
                    }
                },
                
                # Intent distribution
                "intents": dict(self.intent_counts),
                
                # LLM metrics
                "llm": {
                    "total_calls": self.llm_call_count,
                    "cache_hits": self.llm_cache_hits,
                    "cache_misses": self.llm_cache_misses,
                    "cache_hit_rate": self.llm_cache_hits / max(1, self.llm_call_count),
                    "errors": self.llm_errors,
                    "latency_ms": {
                        "min": min(llm_latencies_sorted),
                        "max": max(llm_latencies_sorted),
                        "p50": percentile(llm_latencies_sorted, 0.50),
                        "p95": percentile(llm_latencies_sorted, 0.95),
                    }
                },
                
                # Retrieval metrics
                "retrieval": {
                    "total_searches": self.retrieval_count,
                    "empty_results": self.retrieval_empty_results,
                    "empty_result_rate": self.retrieval_empty_results / max(1, self.retrieval_count),
                    "latency_ms": {
                        "min": min(ret_latencies_sorted),
                        "max": max(ret_latencies_sorted),
                        "p50": percentile(ret_latencies_sorted, 0.50),
                        "p95": percentile(ret_latencies_sorted, 0.95),
                    }
                },
                
                # Cache metrics
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hits / max(1, self.cache_hits + self.cache_misses),
                },
                
                # Circuit breaker metrics
                "circuit_breaker": {
                    "opens": self.circuit_breaker_opens,
                    "failures": self.circuit_breaker_failures,
                }
            }
    
    def reset_metrics(self):
        """Reset all metrics (use with caution)."""
        with self._lock:
            self.request_count = 0
            self.request_latencies.clear()
            self.request_errors = 0
            self.intent_counts.clear()
            self.llm_call_count = 0
            self.llm_cache_hits = 0
            self.llm_cache_misses = 0
            self.llm_latencies.clear()
            self.llm_errors = 0
            self.retrieval_count = 0
            self.retrieval_latencies.clear()
            self.retrieval_empty_results = 0
            self.cache_hits = 0
            self.cache_misses = 0
            self.circuit_breaker_opens = 0
            self.circuit_breaker_failures = 0
            self.start_time = datetime.now()


# Global metrics instance
metrics = MetricsCollector()
