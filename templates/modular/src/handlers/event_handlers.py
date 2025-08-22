"""
Event Handlers
==============
Handle server lifecycle and request/response events.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


async def handle_startup(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handle server startup event.
    
    Args:
        config: Optional startup configuration
    
    Returns:
        Startup status
    """
    try:
        logger.info("Server startup initiated")
        startup_tasks = []
        
        # Initialize subsystems
        startup_tasks.append(("Database", initialize_database()))
        startup_tasks.append(("Cache", initialize_cache_system()))
        startup_tasks.append(("API Client", initialize_api_client()))
        startup_tasks.append(("Monitoring", initialize_monitoring()))
        
        results = {}
        for name, task in startup_tasks:
            try:
                await task
                results[name] = "initialized"
                logger.info(f"{name} initialized successfully")
            except Exception as e:
                results[name] = f"failed: {str(e)}"
                logger.error(f"{name} initialization failed: {e}")
        
        return {
            "type": "startup_complete",
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "status": "ready" if all(v == "initialized" for v in results.values()) else "partial"
        }
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return {
            "type": "startup_failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_shutdown(
    graceful: bool = True,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Handle server shutdown event.
    
    Args:
        graceful: Whether to perform graceful shutdown
        timeout: Shutdown timeout in seconds
    
    Returns:
        Shutdown status
    """
    try:
        logger.info(f"Server shutdown initiated (graceful={graceful})")
        
        if graceful:
            # Graceful shutdown sequence
            shutdown_tasks = []
            
            # Stop accepting new requests
            logger.info("Stopping new request acceptance")
            
            # Wait for pending operations
            logger.info("Waiting for pending operations")
            await asyncio.sleep(1)  # Give time for operations to complete
            
            # Cleanup tasks
            shutdown_tasks.append(("Active connections", close_connections()))
            shutdown_tasks.append(("Cache", cleanup_cache_system()))
            shutdown_tasks.append(("API clients", close_api_clients()))
            shutdown_tasks.append(("Database", close_database()))
            
            results = {}
            for name, task in shutdown_tasks:
                try:
                    await asyncio.wait_for(task, timeout=timeout/len(shutdown_tasks))
                    results[name] = "closed"
                    logger.info(f"{name} closed successfully")
                except asyncio.TimeoutError:
                    results[name] = "timeout"
                    logger.warning(f"{name} shutdown timeout")
                except Exception as e:
                    results[name] = f"error: {str(e)}"
                    logger.error(f"{name} shutdown error: {e}")
            
            return {
                "type": "shutdown_complete",
                "graceful": True,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
        
        else:
            # Force shutdown
            logger.warning("Forcing immediate shutdown")
            return {
                "type": "shutdown_forced",
                "graceful": False,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Shutdown error: {e}")
        return {
            "type": "shutdown_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_request(
    request: Dict[str, Any],
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle incoming requests with middleware processing.
    
    Args:
        request: Request data
        request_id: Optional request identifier
    
    Returns:
        Request processing status
    """
    try:
        request_id = request_id or generate_request_id()
        logger.debug(f"Processing request {request_id}")
        
        # Request validation
        validation_result = await validate_request(request)
        if not validation_result["valid"]:
            return {
                "type": "request_rejected",
                "request_id": request_id,
                "reason": validation_result.get("reason", "Validation failed"),
                "timestamp": datetime.now().isoformat()
            }
        
        # Apply request middleware
        for middleware in _request_middleware:
            request = await middleware(request)
        
        # Track request
        _active_requests[request_id] = {
            "request": request,
            "started_at": datetime.now().isoformat(),
            "status": "processing"
        }
        
        return {
            "type": "request_accepted",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Request handling error: {e}")
        return {
            "type": "request_error",
            "request_id": request_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_response(
    response: Dict[str, Any],
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle outgoing responses with middleware processing.
    
    Args:
        response: Response data
        request_id: Optional request identifier
    
    Returns:
        Response processing status
    """
    try:
        logger.debug(f"Processing response for request {request_id}")
        
        # Apply response middleware
        for middleware in _response_middleware:
            response = await middleware(response)
        
        # Update request tracking
        if request_id and request_id in _active_requests:
            _active_requests[request_id]["status"] = "completed"
            _active_requests[request_id]["completed_at"] = datetime.now().isoformat()
            
            # Calculate processing time
            started = _active_requests[request_id]["started_at"]
            # processing_time = calculate_time_diff(started, datetime.now())
            
            # Clean up after delay
            asyncio.create_task(cleanup_request(request_id, delay=60))
        
        return {
            "type": "response_sent",
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Response handling error: {e}")
        return {
            "type": "response_error",
            "request_id": request_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Helper functions for lifecycle management
async def initialize_database():
    """Initialize database connections."""
    await asyncio.sleep(0.1)  # Simulate initialization
    logger.debug("Database initialized")


async def initialize_cache_system():
    """Initialize cache system."""
    await asyncio.sleep(0.1)  # Simulate initialization
    logger.debug("Cache system initialized")


async def initialize_api_client():
    """Initialize API client."""
    await asyncio.sleep(0.1)  # Simulate initialization
    logger.debug("API client initialized")


async def initialize_monitoring():
    """Initialize monitoring system."""
    await asyncio.sleep(0.1)  # Simulate initialization
    logger.debug("Monitoring initialized")


async def close_connections():
    """Close all active connections."""
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.debug("Connections closed")


async def cleanup_cache_system():
    """Clean up cache system."""
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.debug("Cache cleaned up")


async def close_api_clients():
    """Close API clients."""
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.debug("API clients closed")


async def close_database():
    """Close database connections."""
    await asyncio.sleep(0.1)  # Simulate cleanup
    logger.debug("Database closed")


# Request/Response tracking
_active_requests: Dict[str, Dict[str, Any]] = {}
_request_middleware: list[Callable] = []
_response_middleware: list[Callable] = []


def generate_request_id() -> str:
    """Generate unique request ID."""
    import uuid
    return str(uuid.uuid4())


async def validate_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate incoming request.
    
    Args:
        request: Request to validate
    
    Returns:
        Validation result
    """
    # Basic validation
    if not request:
        return {"valid": False, "reason": "Empty request"}
    
    if "type" not in request:
        return {"valid": False, "reason": "Missing request type"}
    
    return {"valid": True}


async def cleanup_request(request_id: str, delay: int = 0):
    """
    Clean up request tracking after delay.
    
    Args:
        request_id: Request to clean up
        delay: Delay in seconds
    """
    if delay > 0:
        await asyncio.sleep(delay)
    
    if request_id in _active_requests:
        del _active_requests[request_id]
        logger.debug(f"Cleaned up request {request_id}")


def register_request_middleware(middleware: Callable):
    """
    Register request middleware.
    
    Args:
        middleware: Middleware function
    """
    _request_middleware.append(middleware)
    logger.debug(f"Registered request middleware: {middleware.__name__}")


def register_response_middleware(middleware: Callable):
    """
    Register response middleware.
    
    Args:
        middleware: Middleware function
    """
    _response_middleware.append(middleware)
    logger.debug(f"Registered response middleware: {middleware.__name__}")


async def get_active_requests() -> list[Dict[str, Any]]:
    """
    Get list of active requests.
    
    Returns:
        Active requests
    """
    return [
        {
            "request_id": req_id,
            **req_data
        }
        for req_id, req_data in _active_requests.items()
    ]


async def cancel_request(request_id: str) -> bool:
    """
    Cancel an active request.
    
    Args:
        request_id: Request to cancel
    
    Returns:
        True if cancelled successfully
    """
    if request_id in _active_requests:
        _active_requests[request_id]["status"] = "cancelled"
        _active_requests[request_id]["cancelled_at"] = datetime.now().isoformat()
        logger.info(f"Request {request_id} cancelled")
        return True
    return False