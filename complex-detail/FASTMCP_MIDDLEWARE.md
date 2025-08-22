# FastMCP Middleware Documentation

## Table of Contents
- [Overview](#overview)
- [Basic Middleware](#basic-middleware)
- [Request/Response Interception](#requestresponse-interception)
- [Authentication Middleware](#authentication-middleware)
- [Rate Limiting Middleware](#rate-limiting-middleware)
- [Logging Middleware](#logging-middleware)
- [Error Handling Middleware](#error-handling-middleware)
- [Middleware Chaining](#middleware-chaining)
- [Context Propagation](#context-propagation)
- [Custom Middleware](#custom-middleware)
- [Best Practices](#best-practices)

## Overview

FastMCP middleware provides a powerful way to implement cross-cutting concerns across your MCP server. Middleware functions can intercept and modify requests and responses, add authentication, implement rate limiting, log operations, and more.

## Basic Middleware

### Simple Request Logger

```python
from fastmcp import FastMCP
from typing import Callable, Any, Dict
import time
import logging

logger = logging.getLogger(__name__)
mcp = FastMCP("middleware-server")

async def request_logger(request: Dict[str, Any], next: Callable) -> Any:
    """Log all incoming requests."""
    start_time = time.time()
    logger.info(f"Request: {request.get('method', 'unknown')}")
    
    # Call the next middleware or handler
    response = await next(request)
    
    duration = time.time() - start_time
    logger.info(f"Response time: {duration:.2f}s")
    return response

# Register middleware
mcp.add_middleware(request_logger)
```

### Response Modifier

```python
async def response_enricher(request: Dict[str, Any], next: Callable) -> Any:
    """Add metadata to all responses."""
    response = await next(request)
    
    if isinstance(response, dict):
        response["_metadata"] = {
            "server_version": "1.0.0",
            "timestamp": time.time(),
            "request_id": request.get("id")
        }
    
    return response

mcp.add_middleware(response_enricher)
```

## Request/Response Interception

### Request Validation Middleware

```python
from fastmcp.exceptions import MCPError

async def request_validator(request: Dict[str, Any], next: Callable) -> Any:
    """Validate request structure and parameters."""
    
    # Check for required fields
    if "method" not in request:
        raise MCPError("Missing required field: method")
    
    # Validate method format
    method = request["method"]
    if not isinstance(method, str) or not method.startswith("tools/"):
        raise MCPError(f"Invalid method format: {method}")
    
    # Validate parameters
    params = request.get("params", {})
    if not isinstance(params, dict):
        raise MCPError("Parameters must be a dictionary")
    
    # Add validation timestamp
    request["_validated_at"] = time.time()
    
    return await next(request)
```

### Response Transformation Middleware

```python
import json

async def response_transformer(request: Dict[str, Any], next: Callable) -> Any:
    """Transform responses based on client preferences."""
    response = await next(request)
    
    # Check for format preference in request
    format_type = request.get("params", {}).get("_format", "json")
    
    if format_type == "xml" and isinstance(response, dict):
        # Convert to XML (simplified)
        xml_response = dict_to_xml(response)
        return {"content": xml_response, "format": "xml"}
    elif format_type == "compact":
        # Remove null values and empty lists
        return compact_response(response)
    
    return response

def compact_response(data: Any) -> Any:
    """Remove null values and empty collections."""
    if isinstance(data, dict):
        return {k: compact_response(v) for k, v in data.items() 
                if v is not None and v != [] and v != {}}
    elif isinstance(data, list):
        return [compact_response(item) for item in data]
    return data
```

## Authentication Middleware

### Bearer Token Authentication

```python
import os
from fastmcp.exceptions import AuthenticationError

async def bearer_auth_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Authenticate requests using Bearer tokens."""
    
    # Extract authorization header
    headers = request.get("headers", {})
    auth_header = headers.get("Authorization", "")
    
    if not auth_header.startswith("Bearer "):
        raise AuthenticationError("Missing or invalid Bearer token")
    
    token = auth_header[7:]  # Remove "Bearer " prefix
    
    # Validate token
    if not validate_token(token):
        raise AuthenticationError("Invalid token")
    
    # Add user context to request
    user_info = get_user_from_token(token)
    request["user"] = user_info
    
    return await next(request)

def validate_token(token: str) -> bool:
    """Validate token against stored tokens or JWT."""
    valid_tokens = os.getenv("VALID_TOKENS", "").split(",")
    return token in valid_tokens

def get_user_from_token(token: str) -> Dict[str, Any]:
    """Extract user information from token."""
    # In production, decode JWT or lookup in database
    return {
        "id": "user123",
        "email": "user@example.com",
        "permissions": ["read", "write"]
    }
```

### API Key Authentication

```python
async def api_key_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Authenticate using API keys."""
    
    # Check for API key in headers or params
    headers = request.get("headers", {})
    params = request.get("params", {})
    
    api_key = headers.get("X-API-Key") or params.get("api_key")
    
    if not api_key:
        raise AuthenticationError("API key required")
    
    # Validate and get client info
    client_info = validate_api_key(api_key)
    if not client_info:
        raise AuthenticationError("Invalid API key")
    
    # Add client context
    request["client"] = client_info
    
    # Track usage
    await track_api_usage(api_key, request.get("method"))
    
    return await next(request)

async def track_api_usage(api_key: str, method: str):
    """Track API usage for rate limiting and billing."""
    # Implementation depends on your storage backend
    pass
```

## Rate Limiting Middleware

### Token Bucket Rate Limiter

```python
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class TokenBucket:
    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = datetime.now()
        self.lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        async with self.lock:
            await self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    async def _refill(self):
        now = datetime.now()
        elapsed = (now - self.last_refill).total_seconds()
        tokens_to_add = int(elapsed * self.refill_rate)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now

# Store buckets per client
rate_limiters = defaultdict(lambda: TokenBucket(capacity=100, refill_rate=10))

async def rate_limit_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Implement rate limiting per client."""
    
    # Identify client
    client_id = request.get("client", {}).get("id", "anonymous")
    
    # Get rate limiter for client
    limiter = rate_limiters[client_id]
    
    # Try to consume a token
    if not await limiter.consume():
        raise MCPError("Rate limit exceeded", code=429)
    
    # Add rate limit headers to response
    response = await next(request)
    if isinstance(response, dict):
        response["_rate_limit"] = {
            "remaining": int(limiter.tokens),
            "limit": limiter.capacity,
            "reset": (datetime.now() + timedelta(seconds=60)).isoformat()
        }
    
    return response
```

### Sliding Window Rate Limiter

```python
from collections import deque

class SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = deque()
    
    async def allow_request(self) -> bool:
        now = time.time()
        
        # Remove old requests outside window
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        # Check if we can allow this request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False

sliding_limiters = defaultdict(lambda: SlidingWindowLimiter(60, 60))  # 60 requests per minute

async def sliding_rate_limit_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Sliding window rate limiting."""
    client_id = request.get("client", {}).get("id", "anonymous")
    limiter = sliding_limiters[client_id]
    
    if not await limiter.allow_request():
        retry_after = limiter.window_seconds - (time.time() - limiter.requests[0])
        raise MCPError(
            "Rate limit exceeded",
            code=429,
            details={"retry_after": int(retry_after)}
        )
    
    return await next(request)
```

## Logging Middleware

### Comprehensive Request/Response Logger

```python
import json
from datetime import datetime

async def comprehensive_logger(request: Dict[str, Any], next: Callable) -> Any:
    """Log detailed request and response information."""
    
    # Generate request ID if not present
    request_id = request.get("id", str(uuid.uuid4()))
    request["id"] = request_id
    
    # Log request
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "method": request.get("method"),
        "params": request.get("params"),
        "client": request.get("client", {}).get("id"),
        "user": request.get("user", {}).get("email")
    }
    
    logger.info(f"Request: {json.dumps(log_entry)}")
    
    try:
        # Process request
        start_time = time.time()
        response = await next(request)
        duration = time.time() - start_time
        
        # Log successful response
        logger.info(f"Response: {request_id} completed in {duration:.3f}s")
        
        # Add performance metrics
        if isinstance(response, dict):
            response["_performance"] = {
                "duration_ms": duration * 1000,
                "request_id": request_id
            }
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise
```

### Audit Logger

```python
async def audit_logger(request: Dict[str, Any], next: Callable) -> Any:
    """Log security-relevant operations for audit trails."""
    
    method = request.get("method", "")
    
    # Determine if this is an auditable operation
    auditable_methods = ["create", "update", "delete", "grant", "revoke"]
    should_audit = any(op in method.lower() for op in auditable_methods)
    
    if should_audit:
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "user": request.get("user", {}).get("email", "anonymous"),
            "method": method,
            "params": sanitize_sensitive_data(request.get("params", {})),
            "ip_address": request.get("headers", {}).get("X-Forwarded-For"),
            "user_agent": request.get("headers", {}).get("User-Agent")
        }
        
        # Log to audit trail (could be database, file, or external service)
        await log_audit_entry(audit_entry)
    
    return await next(request)

def sanitize_sensitive_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive data from parameters before logging."""
    sensitive_keys = ["password", "token", "secret", "api_key", "ssn"]
    sanitized = {}
    
    for key, value in params.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = value
    
    return sanitized
```

## Error Handling Middleware

### Global Error Handler

```python
async def error_handler_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Handle and transform errors consistently."""
    try:
        return await next(request)
    except MCPError as e:
        # MCP-specific errors pass through with formatting
        logger.error(f"MCP Error: {e.message}")
        return {
            "error": {
                "code": e.code,
                "message": e.message,
                "details": e.details
            }
        }
    except ValidationError as e:
        # Validation errors
        logger.warning(f"Validation error: {str(e)}")
        return {
            "error": {
                "code": 400,
                "message": "Validation failed",
                "details": str(e)
            }
        }
    except AuthenticationError as e:
        # Authentication failures
        logger.warning(f"Auth failed: {str(e)}")
        return {
            "error": {
                "code": 401,
                "message": "Authentication required",
                "details": str(e)
            }
        }
    except Exception as e:
        # Unexpected errors
        logger.exception("Unexpected error in request processing")
        
        # In production, don't expose internal errors
        if os.getenv("ENVIRONMENT") == "production":
            return {
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "request_id": request.get("id")
                }
            }
        else:
            return {
                "error": {
                    "code": 500,
                    "message": str(e),
                    "type": type(e).__name__,
                    "request_id": request.get("id")
                }
            }
```

### Retry Middleware

```python
import asyncio
from typing import List

async def retry_middleware(
    request: Dict[str, Any], 
    next: Callable,
    max_retries: int = 3,
    retry_on: List[type] = None
) -> Any:
    """Automatically retry failed requests."""
    
    retry_on = retry_on or [ConnectionError, TimeoutError]
    attempt = 0
    last_error = None
    
    while attempt < max_retries:
        try:
            return await next(request)
        except Exception as e:
            if not any(isinstance(e, error_type) for error_type in retry_on):
                # Don't retry this type of error
                raise
            
            attempt += 1
            last_error = e
            
            if attempt < max_retries:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
                await asyncio.sleep(wait_time)
    
    # All retries exhausted
    raise last_error
```

## Middleware Chaining

### Proper Middleware Order

```python
from fastmcp import FastMCP

mcp = FastMCP("chained-server")

# Order matters! Add middleware from outermost to innermost:

# 1. Error handling (outermost - catches all errors)
mcp.add_middleware(error_handler_middleware)

# 2. Logging (logs all requests including errors)
mcp.add_middleware(comprehensive_logger)

# 3. Rate limiting (before auth to prevent brute force)
mcp.add_middleware(rate_limit_middleware)

# 4. Authentication (validates user)
mcp.add_middleware(bearer_auth_middleware)

# 5. Authorization (checks permissions)
mcp.add_middleware(authorization_middleware)

# 6. Validation (validates request structure)
mcp.add_middleware(request_validator)

# 7. Caching (closest to handlers)
mcp.add_middleware(cache_middleware)
```

### Conditional Middleware

```python
async def conditional_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Apply middleware conditionally based on request."""
    
    method = request.get("method", "")
    
    # Skip authentication for public methods
    public_methods = ["tools/list", "resources/list", "health"]
    if method in public_methods:
        return await next(request)
    
    # Apply authentication for protected methods
    return await bearer_auth_middleware(request, next)
```

### Middleware Factory

```python
def create_middleware(config: Dict[str, Any]) -> Callable:
    """Factory to create configured middleware."""
    
    async def configured_middleware(request: Dict[str, Any], next: Callable) -> Any:
        # Use configuration
        if config.get("log_requests"):
            logger.info(f"Request: {request.get('method')}")
        
        if config.get("add_timestamp"):
            request["timestamp"] = time.time()
        
        response = await next(request)
        
        if config.get("add_server_info"):
            if isinstance(response, dict):
                response["server"] = config.get("server_name", "unknown")
        
        return response
    
    return configured_middleware

# Create and use configured middleware
mcp.add_middleware(create_middleware({
    "log_requests": True,
    "add_timestamp": True,
    "add_server_info": True,
    "server_name": "production-server-1"
}))
```

## Context Propagation

### Request Context Middleware

```python
import contextvars

# Create context variables
request_context = contextvars.ContextVar("request_context")
user_context = contextvars.ContextVar("user_context")

async def context_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Set up context for request processing."""
    
    # Set request context
    request_token = request_context.set({
        "request_id": request.get("id"),
        "method": request.get("method"),
        "timestamp": time.time()
    })
    
    # Set user context if authenticated
    user_token = None
    if "user" in request:
        user_token = user_context.set(request["user"])
    
    try:
        return await next(request)
    finally:
        # Clean up context
        request_context.reset(request_token)
        if user_token:
            user_context.reset(user_token)

# Use context in tools
@mcp.tool()
async def context_aware_tool():
    """Tool that uses request context."""
    ctx = request_context.get()
    user = user_context.get(None)
    
    return {
        "request_id": ctx.get("request_id"),
        "user": user.get("email") if user else "anonymous",
        "processing_time": time.time() - ctx.get("timestamp")
    }
```

### Distributed Tracing

```python
import uuid

async def tracing_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Add distributed tracing support."""
    
    # Extract or create trace ID
    headers = request.get("headers", {})
    trace_id = headers.get("X-Trace-ID", str(uuid.uuid4()))
    span_id = str(uuid.uuid4())
    
    # Add to request
    request["trace"] = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": headers.get("X-Parent-Span-ID")
    }
    
    # Log trace start
    logger.info(f"Trace start: {trace_id}/{span_id}")
    
    try:
        response = await next(request)
        
        # Add trace to response
        if isinstance(response, dict):
            response["_trace"] = {
                "trace_id": trace_id,
                "span_id": span_id
            }
        
        logger.info(f"Trace complete: {trace_id}/{span_id}")
        return response
        
    except Exception as e:
        logger.error(f"Trace failed: {trace_id}/{span_id}: {str(e)}")
        raise
```

## Custom Middleware

### Cache Middleware

```python
from functools import lru_cache
import hashlib

cache_store = {}

async def cache_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Cache responses for read operations."""
    
    method = request.get("method", "")
    
    # Only cache read operations
    if not any(op in method for op in ["get", "list", "read", "fetch"]):
        return await next(request)
    
    # Generate cache key
    cache_key = generate_cache_key(request)
    
    # Check cache
    if cache_key in cache_store:
        cached = cache_store[cache_key]
        if time.time() - cached["timestamp"] < 300:  # 5 minute TTL
            logger.info(f"Cache hit: {cache_key}")
            return cached["response"]
    
    # Process request
    response = await next(request)
    
    # Store in cache
    cache_store[cache_key] = {
        "response": response,
        "timestamp": time.time()
    }
    
    return response

def generate_cache_key(request: Dict[str, Any]) -> str:
    """Generate cache key from request."""
    key_parts = [
        request.get("method", ""),
        json.dumps(request.get("params", {}), sort_keys=True)
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()
```

### Metrics Collection Middleware

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
request_count = Counter('mcp_requests_total', 'Total MCP requests', ['method', 'status'])
request_duration = Histogram('mcp_request_duration_seconds', 'Request duration', ['method'])
active_requests = Gauge('mcp_active_requests', 'Active requests')

async def metrics_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Collect metrics for monitoring."""
    
    method = request.get("method", "unknown")
    active_requests.inc()
    
    with request_duration.labels(method=method).time():
        try:
            response = await next(request)
            request_count.labels(method=method, status="success").inc()
            return response
        except Exception as e:
            request_count.labels(method=method, status="error").inc()
            raise
        finally:
            active_requests.dec()
```

## Best Practices

### 1. Middleware Design Principles

```python
# Good: Single Responsibility
async def auth_middleware(request, next):
    """Only handles authentication."""
    # Authenticate
    return await next(request)

async def logging_middleware(request, next):
    """Only handles logging."""
    # Log
    return await next(request)

# Bad: Multiple Responsibilities
async def everything_middleware(request, next):
    """Does too much - hard to maintain and test."""
    # Authenticate
    # Log
    # Rate limit
    # Cache
    # Transform
    return await next(request)
```

### 2. Error Handling

```python
async def safe_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Properly handle errors without breaking the chain."""
    try:
        return await next(request)
    except SpecificError as e:
        # Handle specific errors
        logger.warning(f"Handled error: {e}")
        # Decide whether to continue or return error response
        return {"error": str(e)}
    except Exception as e:
        # Don't swallow unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise  # Let error handler middleware deal with it
```

### 3. Performance Considerations

```python
async def efficient_middleware(request: Dict[str, Any], next: Callable) -> Any:
    """Minimize overhead in middleware."""
    
    # Quick checks first
    if should_skip(request):
        return await next(request)
    
    # Avoid blocking operations
    # Bad: time.sleep(1)
    # Good: await asyncio.sleep(1)
    
    # Use async operations
    # Bad: requests.get(url)
    # Good: async with aiohttp.ClientSession() as session:
    #         await session.get(url)
    
    return await next(request)
```

### 4. Testing Middleware

```python
import pytest

@pytest.mark.asyncio
async def test_auth_middleware():
    """Test authentication middleware."""
    
    # Mock next handler
    async def mock_handler(request):
        return {"success": True}
    
    # Test with valid token
    request = {
        "headers": {"Authorization": "Bearer valid-token"},
        "method": "tools/test"
    }
    response = await bearer_auth_middleware(request, mock_handler)
    assert response["success"] == True
    
    # Test with invalid token
    request = {
        "headers": {"Authorization": "Bearer invalid-token"},
        "method": "tools/test"
    }
    with pytest.raises(AuthenticationError):
        await bearer_auth_middleware(request, mock_handler)
```

### 5. Configuration

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class MiddlewareConfig:
    """Configuration for middleware stack."""
    enable_auth: bool = True
    enable_rate_limit: bool = True
    enable_cache: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    cache_ttl: int = 300
    log_level: str = "INFO"

def setup_middleware(mcp: FastMCP, config: MiddlewareConfig):
    """Configure middleware based on settings."""
    
    # Always add error handling
    mcp.add_middleware(error_handler_middleware)
    
    # Conditional middleware
    if config.log_level != "NONE":
        mcp.add_middleware(comprehensive_logger)
    
    if config.enable_rate_limit:
        limiter = create_rate_limiter(
            config.rate_limit_requests,
            config.rate_limit_window
        )
        mcp.add_middleware(limiter)
    
    if config.enable_auth:
        mcp.add_middleware(bearer_auth_middleware)
    
    if config.enable_cache:
        cache = create_cache_middleware(config.cache_ttl)
        mcp.add_middleware(cache)

# Usage
config = MiddlewareConfig(
    enable_auth=True,
    enable_cache=True,
    rate_limit_requests=200
)
setup_middleware(mcp, config)
```

## Advanced Patterns

### Middleware Composition

```python
def compose_middleware(*middlewares):
    """Compose multiple middleware into one."""
    async def composed(request: Dict[str, Any], next: Callable) -> Any:
        # Build the chain
        async def chain(idx: int, req: Dict[str, Any]) -> Any:
            if idx >= len(middlewares):
                return await next(req)
            return await middlewares[idx](req, lambda r: chain(idx + 1, r))
        
        return await chain(0, request)
    
    return composed

# Compose auth and logging
auth_and_logging = compose_middleware(
    logging_middleware,
    auth_middleware
)
mcp.add_middleware(auth_and_logging)
```

### Dynamic Middleware Loading

```python
import importlib

def load_middleware_from_config(config_path: str) -> List[Callable]:
    """Load middleware dynamically from configuration."""
    with open(config_path) as f:
        config = json.load(f)
    
    middlewares = []
    for middleware_config in config["middlewares"]:
        module_name = middleware_config["module"]
        function_name = middleware_config["function"]
        params = middleware_config.get("params", {})
        
        # Import module
        module = importlib.import_module(module_name)
        middleware_func = getattr(module, function_name)
        
        # Create configured middleware
        if params:
            middleware = middleware_func(**params)
        else:
            middleware = middleware_func
        
        middlewares.append(middleware)
    
    return middlewares

# Load and apply
middlewares = load_middleware_from_config("middleware.json")
for middleware in middlewares:
    mcp.add_middleware(middleware)
```

## Summary

FastMCP middleware provides a powerful and flexible way to implement cross-cutting concerns in your MCP server. Key points:

1. **Order Matters**: Add middleware from outermost to innermost
2. **Single Responsibility**: Each middleware should do one thing well
3. **Error Handling**: Always handle errors appropriately
4. **Performance**: Keep middleware lightweight and async
5. **Testing**: Test middleware in isolation
6. **Configuration**: Make middleware configurable
7. **Composition**: Combine simple middleware for complex behavior

Middleware enables clean separation of concerns, making your MCP server more maintainable, testable, and scalable.