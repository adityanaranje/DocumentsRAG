"""
Circuit breaker pattern for external dependencies.
"""
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional
from config import config


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker for external dependencies.
    
    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Failure threshold reached, all requests rejected
    - HALF_OPEN: After timeout, allow test requests
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = None,
        timeout_seconds: int = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Name of the circuit (for logging)
            failure_threshold: Number of consecutive failures before opening
            timeout_seconds: Seconds to wait before attempting recovery
        """
        self.name = name
        self.failure_threshold = failure_threshold or config.CIRCUIT_BREAKER_FAILURE_THRESHOLD
        self.timeout = timeout_seconds or config.CIRCUIT_BREAKER_TIMEOUT
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerError: If circuit is open
        """
        with self._lock:
            # Check state transitions
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Will retry after {self.timeout}s"
                    )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                # After 3 successful calls in HALF_OPEN, close the circuit
                if self.success_count >= 3:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # In HALF_OPEN, any failure immediately opens circuit
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.failure_count = 0
                return
            
            # In CLOSED, open after threshold
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
    
    def get_state(self) -> dict:
        """Get current state for monitoring."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": self.last_failure_time
            }


class CircuitBreakerManager:
    """Manages circuit breakers for different dependencies."""
    
    def __init__(self):
        self.breakers = {
            "llm": CircuitBreaker("llm", failure_threshold=5, timeout_seconds=60),
            "retriever": CircuitBreaker("retriever", failure_threshold=3, timeout_seconds=30),
            "vector_store": CircuitBreaker("vector_store", failure_threshold=3, timeout_seconds=30),
        }
    
    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get circuit breaker by name."""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name)
        return self.breakers[name]
    
    def get_all_states(self) -> dict:
        """Get states of all circuit breakers."""
        return {
            name: breaker.get_state()
            for name, breaker in self.breakers.items()
        }
    
    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()
