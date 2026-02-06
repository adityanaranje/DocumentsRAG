"""
Monitoring and health check endpoints.
"""
from flask import Blueprint, jsonify
import os
from datetime import datetime
from utils.metrics import metrics
from utils.request_logger import request_logger
from utils.cache import cache_manager
from utils.circuit_breaker import circuit_breaker_manager
from config import config

monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api')


@monitoring_bp.route('/health', methods=['GET'])
def health_check():
    """
    Basic health check endpoint.
    Returns 200 if service is running.
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": config.VERSION,
        "environment": config.ENVIRONMENT.value
    }), 200


@monitoring_bp.route('/ready', methods=['GET'])
def readiness_check():
    """
    Readiness check - validates critical dependencies.
    Returns 200 if all dependencies are available.
    """
    checks = {}
    overall_ready = True
    
    # Check vector store
    try:
        vector_store_path = config.VECTOR_STORE_PATH
        if os.path.exists(vector_store_path):
            checks["vector_store"] = "ready"
        else:
            checks["vector_store"] = "not_found"
            overall_ready = False
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"
        overall_ready = False
    
    # Check LLM API key
    if config.GROQ_API_KEY:
        checks["llm_api"] = "configured"
    else:
        checks["llm_api"] = "missing_api_key"
        overall_ready = False
    
    # Check circuit breakers
    breaker_states = circuit_breaker_manager.get_all_states()
    open_breakers = [name for name, state in breaker_states.items() if state["state"] == "open"]
    
    if open_breakers:
        checks["circuit_breakers"] = f"open: {', '.join(open_breakers)}"
        overall_ready = False
    else:
        checks["circuit_breakers"] = "all_closed"
    
    status_code = 200 if overall_ready else 503
    
    return jsonify({
        "ready": overall_ready,
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }), status_code


@monitoring_bp.route('/metrics', methods=['GET'])
def get_metrics():
    """
    Get application metrics in JSON format.
    """
    if not config.ENABLE_METRICS:
        return jsonify({"error": "Metrics disabled"}), 403
    
    app_metrics = metrics.get_metrics()
    cache_stats = cache_manager.get_all_stats()
    circuit_states = circuit_breaker_manager.get_all_states()
    
    return jsonify({
        "application": app_metrics,
        "cache": cache_stats,
        "circuit_breakers": circuit_states
    }), 200


@monitoring_bp.route('/stats', methods=['GET'])
def get_stats():
    """
    Get human-readable statistics.
    """
    app_metrics = metrics.get_metrics()
    
    # Get additional stats from request logger
    recent_requests = request_logger.get_recent_requests(limit=10)
    intent_dist = request_logger.get_intent_distribution(hours=24)
    error_rate_24h = request_logger.get_error_rate(hours=24)
    
    return jsonify({
        "summary": {
            "total_requests": app_metrics["requests"]["total"],
            "active_requests": app_metrics["requests"]["active"],
            "error_rate": app_metrics["requests"]["error_rate"],
            "avg_latency_ms": app_metrics["requests"]["latency_ms"]["p50"],
            "uptime_hours": app_metrics["uptime_seconds"] / 3600,
        },
        "intent_distribution": intent_dist,
        "recent_requests": recent_requests,
        "error_rate_24h": error_rate_24h,
        "cache_performance": {
            "llm_cache_hit_rate": app_metrics["llm"]["cache_hit_rate"],
            "app_cache_hit_rate": app_metrics["cache"]["hit_rate"],
        }
    }), 200


@monitoring_bp.route('/logs/recent', methods=['GET'])
def get_recent_logs():
    """
    Get recent request logs.
    """
    limit = int(request.args.get('limit', 50))
    limit = min(limit, 500)  # Cap at 500
    
    recent = request_logger.get_recent_requests(limit=limit)
    
    return jsonify({
        "count": len(recent),
        "requests": recent
    }), 200
