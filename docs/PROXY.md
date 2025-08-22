# FastMCP Proxy Server Documentation

Complete guide for FastMCP's proxy server functionality, enabling load balancing, authentication, caching, and routing across multiple MCP servers.

## Table of Contents
1. [Overview](#overview)
2. [Basic Setup](#basic-setup)
3. [Server Configuration](#server-configuration)
4. [Load Balancing](#load-balancing)
5. [Authentication](#authentication)
6. [Caching](#caching)
7. [Rate Limiting](#rate-limiting)
8. [Monitoring & Metrics](#monitoring--metrics)
9. [Advanced Patterns](#advanced-patterns)
10. [Production Configuration](#production-configuration)

## Overview

FastMCP Proxy acts as a gateway to multiple MCP servers, providing:
- **Load balancing** across multiple backend servers
- **Authentication** and authorization
- **Caching** for improved performance
- **Rate limiting** for API protection
- **Request routing** based on paths or headers
- **Health checking** and failover
- **Monitoring** and metrics collection
- **Transport translation** (HTTP ↔ STDIO)

## Basic Setup

### Simple Proxy Configuration

```python
from fastmcp.proxy import MCPProxy

# Basic proxy with two backend servers
proxy = MCPProxy(
    servers=[
        "http://server1.example.com:8001",
        "http://server2.example.com:8002"
    ]
)

if __name__ == "__main__":
    proxy.run(host="0.0.0.0", port=8000)
```

### With Server Configuration

```python
from fastmcp.proxy import MCPProxy, ServerConfig

proxy = MCPProxy(
    servers=[
        ServerConfig(
            url="http://server1:8001",
            name="primary",
            weight=3  # Gets 3x more traffic
        ),
        ServerConfig(
            url="http://server2:8002",
            name="secondary",
            weight=1
        )
    ]
)
```

## Server Configuration

### Detailed Server Setup

```python
from fastmcp.proxy import ServerConfig, HTTPHealthCheck

servers = [
    ServerConfig(
        name="user-service",
        url="http://user-service:8001",
        
        # Routing
        prefix="/users",  # Route /users/* to this server
        
        # Load balancing
        weight=2,  # Relative weight for load balancing
        
        # Health checking
        health_check=HTTPHealthCheck(
            endpoint="/health",
            interval=30,  # Check every 30 seconds
            timeout=5,
            healthy_threshold=2,  # 2 successes to mark healthy
            unhealthy_threshold=3  # 3 failures to mark unhealthy
        ),
        
        # Connection settings
        max_connections=50,
        connection_timeout=10,
        
        # Retry policy
        retry_attempts=3,
        retry_backoff=1.0,
        
        # Metadata
        tags=["production", "critical"],
        metadata={"region": "us-east-1"}
    ),
    
    ServerConfig(
        name="data-service",
        url="http://data-service:8002",
        prefix="/data",
        weight=3,
        health_check=HTTPHealthCheck("/health"),
        max_connections=100
    )
]

proxy = MCPProxy(servers=servers)
```

### Transport Configuration

```python
# Mix different transport types
servers = [
    # HTTP server
    ServerConfig(
        name="http-server",
        url="http://api.example.com:8000",
        transport="http"
    ),
    
    # STDIO server (local process)
    ServerConfig(
        name="stdio-server",
        command="python",
        args=["local_server.py"],
        transport="stdio"
    ),
    
    # WebSocket server
    ServerConfig(
        name="ws-server",
        url="ws://realtime.example.com:8080",
        transport="websocket"
    )
]

# Proxy translates between transports automatically
proxy = MCPProxy(servers=servers)
```

## Load Balancing

### Round Robin (Default)

```python
from fastmcp.proxy import MCPProxy, RoundRobin

proxy = MCPProxy(
    servers=servers,
    load_balancer=RoundRobin()
)
```

### Weighted Round Robin

```python
from fastmcp.proxy import WeightedRoundRobin

proxy = MCPProxy(
    servers=[
        ServerConfig(url="http://server1", weight=3),
        ServerConfig(url="http://server2", weight=2),
        ServerConfig(url="http://server3", weight=1)
    ],
    load_balancer=WeightedRoundRobin()
)
```

### Least Connections

```python
from fastmcp.proxy import LeastConnections

proxy = MCPProxy(
    servers=servers,
    load_balancer=LeastConnections()
)
```

### Random Selection

```python
from fastmcp.proxy import RandomBalancer

proxy = MCPProxy(
    servers=servers,
    load_balancer=RandomBalancer()
)
```

### IP Hash (Sticky Sessions)

```python
from fastmcp.proxy import IPHash

proxy = MCPProxy(
    servers=servers,
    load_balancer=IPHash()  # Same client IP always goes to same server
)
```

### Custom Load Balancer

```python
from fastmcp.proxy import LoadBalancer

class CustomBalancer(LoadBalancer):
    def select_server(self, request, servers):
        """Custom server selection logic."""
        # Example: Route based on header
        if request.headers.get("X-Priority") == "high":
            # Return first available high-priority server
            for server in servers:
                if "high-priority" in server.tags and server.is_healthy:
                    return server
        
        # Default to round-robin
        return servers[self.counter % len(servers)]

proxy = MCPProxy(
    servers=servers,
    load_balancer=CustomBalancer()
)
```

## Authentication

### Basic Authentication

```python
from fastmcp.proxy import BasicAuth

auth = BasicAuth(
    users={
        "admin": "secret123",
        "user": "password456"
    }
)

proxy = MCPProxy(servers=servers, auth=auth)
```

### Bearer Token Authentication

```python
from fastmcp.proxy import BearerAuth

auth = BearerAuth(
    tokens={
        "token123": {"user": "admin", "permissions": ["read", "write"]},
        "token456": {"user": "user", "permissions": ["read"]}
    }
)

proxy = MCPProxy(servers=servers, auth=auth)
```

### JWT Authentication

```python
from fastmcp.proxy import JWTAuth

auth = JWTAuth(
    secret_key="your-secret-key",
    algorithm="HS256",
    issuer="your-issuer",
    audience="your-audience"
)

proxy = MCPProxy(servers=servers, auth=auth)
```

### OAuth 2.0 Integration

```python
from fastmcp.proxy import OAuth2Auth

auth = OAuth2Auth(
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_endpoint="https://auth.example.com/token",
    introspection_endpoint="https://auth.example.com/introspect"
)

proxy = MCPProxy(servers=servers, auth=auth)
```

### Custom Authentication

```python
from fastmcp.proxy import AuthProvider

class APIKeyAuth(AuthProvider):
    def __init__(self, valid_keys):
        self.valid_keys = valid_keys
    
    async def authenticate(self, request):
        """Validate API key from header."""
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            raise AuthenticationError("Missing API key")
        
        if api_key not in self.valid_keys:
            raise AuthenticationError("Invalid API key")
        
        # Return user context
        return {
            "api_key": api_key,
            "permissions": self.valid_keys[api_key]
        }

auth = APIKeyAuth({
    "key123": ["read", "write"],
    "key456": ["read"]
})

proxy = MCPProxy(servers=servers, auth=auth)
```

## Caching

### In-Memory Cache

```python
from fastmcp.proxy import InMemoryCache

cache = InMemoryCache(
    max_size=1000,  # Maximum number of entries
    ttl=300  # Time to live in seconds
)

proxy = MCPProxy(servers=servers, cache=cache)
```

### Redis Cache

```python
from fastmcp.proxy import RedisCache

cache = RedisCache(
    url="redis://localhost:6379",
    ttl=600,
    prefix="mcp_proxy"
)

proxy = MCPProxy(servers=servers, cache=cache)
```

### Cache Configuration

```python
from fastmcp.proxy import CacheConfig

cache_config = CacheConfig(
    # What to cache
    cache_tools=True,  # Cache tool responses
    cache_resources=True,  # Cache resource responses
    cache_prompts=False,  # Don't cache prompts
    
    # Cache keys
    include_headers=["X-User-ID"],  # Include in cache key
    
    # Cache control
    respect_cache_control=True,  # Honor Cache-Control headers
    
    # Patterns
    cache_patterns=[
        r"/api/v1/users/.*",  # Cache user endpoints
        r"/api/v1/products/.*"  # Cache product endpoints
    ],
    exclude_patterns=[
        r"/api/v1/admin/.*"  # Don't cache admin endpoints
    ]
)

proxy = MCPProxy(
    servers=servers,
    cache=cache,
    cache_config=cache_config
)
```

### Custom Cache Strategy

```python
class SmartCache:
    def __init__(self):
        self.cache = {}
    
    async def get(self, key, request):
        """Get from cache with custom logic."""
        # Don't cache if user is admin
        if request.headers.get("X-Role") == "admin":
            return None
        
        return self.cache.get(key)
    
    async def set(self, key, value, request, response):
        """Set in cache with custom logic."""
        # Only cache successful responses
        if response.status_code == 200:
            # Cache for different durations based on endpoint
            if "/static/" in request.path:
                ttl = 3600  # 1 hour for static content
            else:
                ttl = 60  # 1 minute for dynamic content
            
            self.cache[key] = (value, time.time() + ttl)

proxy = MCPProxy(servers=servers, cache=SmartCache())
```

## Rate Limiting

### Basic Rate Limiting

```python
from fastmcp.proxy import RateLimiter, TokenBucket

# 100 requests per minute
rate_limiter = RateLimiter(
    algorithm=TokenBucket(
        capacity=100,
        refill_rate=100/60  # Per second
    )
)

proxy = MCPProxy(servers=servers, rate_limiter=rate_limiter)
```

### Per-Client Rate Limiting

```python
from fastmcp.proxy import PerClientRateLimiter

rate_limiter = PerClientRateLimiter(
    identify_client=lambda request: request.headers.get("X-Client-ID"),
    default_limit=TokenBucket(100, 100/60),
    
    # Custom limits for specific clients
    client_limits={
        "premium-client": TokenBucket(1000, 1000/60),
        "basic-client": TokenBucket(50, 50/60)
    }
)

proxy = MCPProxy(servers=servers, rate_limiter=rate_limiter)
```

### Advanced Rate Limiting

```python
from fastmcp.proxy import RateLimit, HeaderMatcher, PathMatcher

proxy = MCPProxy(
    servers=servers,
    rate_limiter=rate_limiter,
    rate_limit_rules=[
        # Different limits for different clients
        RateLimit(
            matcher=HeaderMatcher("X-Client-Type", "premium"),
            limit=TokenBucket(capacity=1000, refill_rate=100)
        ),
        RateLimit(
            matcher=HeaderMatcher("X-Client-Type", "basic"),
            limit=TokenBucket(capacity=100, refill_rate=10)
        ),
        
        # Limit expensive endpoints more strictly
        RateLimit(
            matcher=PathMatcher("/expensive/*"),
            limit=TokenBucket(capacity=10, refill_rate=1)
        ),
        
        # Combine matchers
        RateLimit(
            matcher=lambda req: (
                req.headers.get("X-Client-Type") == "trial" and
                "/api/v2/" in req.path
            ),
            limit=TokenBucket(capacity=20, refill_rate=2)
        )
    ]
)
```

### Distributed Rate Limiting

```python
from fastmcp.proxy import RedisRateLimiter

# Share rate limits across multiple proxy instances
rate_limiter = RedisRateLimiter(
    redis_url="redis://localhost:6379",
    key_prefix="rate_limit",
    algorithm=TokenBucket(100, 10)
)

proxy = MCPProxy(servers=servers, rate_limiter=rate_limiter)
```

## Monitoring & Metrics

### Prometheus Metrics

```python
from fastmcp.proxy.monitoring import ProxyMetrics
from prometheus_client import start_http_server

# Enable Prometheus metrics
metrics = ProxyMetrics()
proxy = MCPProxy(servers=servers, metrics=metrics)

# Start metrics server
start_http_server(9090)  # Metrics available at :9090/metrics
```

### Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
request_counter = Counter(
    'mcp_proxy_requests_total',
    'Total MCP proxy requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'mcp_proxy_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'mcp_proxy_active_connections',
    'Number of active connections',
    ['server']
)

@proxy.middleware()
async def metrics_middleware(request, next):
    """Custom metrics collection."""
    start_time = time.time()
    server = None
    
    try:
        # Track active connections
        server = request.selected_server
        active_connections.labels(server=server.name).inc()
        
        # Process request
        response = await next(request)
        
        # Record metrics
        duration = time.time() - start_time
        request_counter.labels(
            method=request.method,
            endpoint=request.path,
            status=response.status_code
        ).inc()
        
        request_duration.labels(
            method=request.method,
            endpoint=request.path
        ).observe(duration)
        
        return response
        
    finally:
        if server:
            active_connections.labels(server=server.name).dec()
```

### Logging Configuration

```python
import logging
from fastmcp.proxy import LoggingConfig

logging_config = LoggingConfig(
    level="INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    
    # Log to file
    file="proxy.log",
    max_bytes=10485760,  # 10MB
    backup_count=5,
    
    # Structured logging
    json_format=True,
    
    # What to log
    log_requests=True,
    log_responses=True,
    log_errors=True,
    log_health_checks=False  # Don't log health checks
)

proxy = MCPProxy(
    servers=servers,
    logging_config=logging_config
)
```

## Advanced Patterns

### Request/Response Transformation

```python
@proxy.middleware()
async def transform_middleware(request, next):
    """Transform requests and responses."""
    
    # Modify request
    request.headers["X-Proxy-Version"] = "1.0"
    request.headers["X-Request-ID"] = generate_request_id()
    
    # Add authentication token if missing
    if not request.headers.get("Authorization"):
        request.headers["Authorization"] = f"Bearer {get_token()}"
    
    # Process request
    response = await next(request)
    
    # Modify response
    response.headers["X-Proxy-Server"] = "FastMCP"
    response.headers["X-Response-Time"] = str(time.time())
    
    # Transform response body
    if response.content_type == "application/json":
        data = response.json()
        data["_metadata"] = {
            "proxy_version": "1.0",
            "timestamp": time.time()
        }
        response.body = json.dumps(data)
    
    return response
```

### Circuit Breaker Pattern

```python
from fastmcp.proxy import CircuitBreaker

circuit_breaker = CircuitBreaker(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=60,  # Try again after 60 seconds
    expected_exception=httpx.RequestError
)

@proxy.middleware()
async def circuit_breaker_middleware(request, next):
    """Implement circuit breaker pattern."""
    server = request.selected_server
    breaker = circuit_breakers.get(server.name)
    
    if breaker.is_open:
        # Circuit is open, fail fast
        return ErrorResponse("Service temporarily unavailable", 503)
    
    try:
        response = await next(request)
        breaker.record_success()
        return response
        
    except Exception as e:
        breaker.record_failure()
        if breaker.is_open:
            # Just opened, log it
            logger.error(f"Circuit breaker opened for {server.name}")
        raise
```

### A/B Testing

```python
import random

@proxy.middleware()
async def ab_testing_middleware(request, next):
    """Route requests for A/B testing."""
    
    # Determine test group
    user_id = request.headers.get("X-User-ID")
    if user_id:
        # Consistent routing for same user
        test_group = hash(user_id) % 100
    else:
        # Random for anonymous
        test_group = random.randint(0, 99)
    
    # Route to different servers based on test group
    if test_group < 10:  # 10% to new version
        request.preferred_server = "v2-server"
        request.headers["X-Test-Group"] = "B"
    else:  # 90% to stable version
        request.preferred_server = "v1-server"
        request.headers["X-Test-Group"] = "A"
    
    response = await next(request)
    
    # Add test group to response
    response.headers["X-Test-Group"] = request.headers["X-Test-Group"]
    
    return response
```

### Retry with Backoff

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@proxy.middleware()
async def retry_middleware(request, next):
    """Retry failed requests with exponential backoff."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(httpx.RequestError)
    )
    async def attempt_request():
        return await next(request)
    
    try:
        return await attempt_request()
    except Exception as e:
        # All retries failed
        logger.error(f"Request failed after retries: {e}")
        return ErrorResponse("Service unavailable", 503)
```

## Production Configuration

### Complete Production Setup

```python
from fastmcp.proxy import MCPProxy, ProxyConfig
import os

# Comprehensive production configuration
config = ProxyConfig(
    # Servers
    servers=[
        ServerConfig(
            name="api-v1",
            url=os.getenv("API_V1_URL"),
            prefix="/api/v1",
            weight=3,
            health_check=HTTPHealthCheck(
                endpoint="/health",
                interval=30,
                timeout=5
            ),
            max_connections=100,
            tags=["production", "v1"]
        ),
        ServerConfig(
            name="api-v2",
            url=os.getenv("API_V2_URL"),
            prefix="/api/v2",
            weight=1,  # Less traffic to new version
            health_check=HTTPHealthCheck("/health"),
            max_connections=50,
            tags=["production", "v2", "beta"]
        )
    ],
    
    # Load balancing
    load_balancer=WeightedRoundRobin(),
    
    # Authentication
    auth=JWTAuth(
        secret_key=os.getenv("JWT_SECRET"),
        issuer="fastmcp-proxy",
        audience="api-gateway"
    ),
    
    # Caching
    cache=RedisCache(
        url=os.getenv("REDIS_URL"),
        ttl=300,
        prefix="mcp_proxy"
    ),
    
    # Rate limiting
    rate_limiter=RedisRateLimiter(
        redis_url=os.getenv("REDIS_URL"),
        algorithm=TokenBucket(
            capacity=1000,
            refill_rate=100
        )
    ),
    
    # Monitoring
    metrics=ProxyMetrics(),
    
    # Timeouts
    upstream_timeout=30,
    connect_timeout=5,
    
    # Retry policy
    retry_attempts=3,
    retry_backoff=1.0,
    retry_on_status=[502, 503, 504],
    
    # Security
    allowed_hosts=["api.example.com", "*.example.com"],
    cors_enabled=True,
    cors_origins=["https://app.example.com"],
    cors_credentials=True,
    
    # Request limits
    max_request_size=10 * 1024 * 1024,  # 10MB
    max_header_size=8192,
    
    # Connection pooling
    connection_pool_size=100,
    keepalive_timeout=30,
    
    # Logging
    log_level="INFO",
    log_file="/var/log/mcp-proxy/proxy.log",
    
    # Error handling
    error_handler=custom_error_handler,
    
    # Middleware
    middleware=[
        logging_middleware,
        metrics_middleware,
        auth_middleware,
        rate_limit_middleware,
        cache_middleware
    ]
)

# Create and run proxy
proxy = MCPProxy(config)

if __name__ == "__main__":
    # Production server
    import uvicorn
    uvicorn.run(
        proxy.app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        loop="uvloop",
        log_config=logging_config
    )
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy proxy code
COPY proxy.py .
COPY config.py .

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run proxy
CMD ["python", "proxy.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-proxy
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-proxy
  template:
    metadata:
      labels:
        app: mcp-proxy
    spec:
      containers:
      - name: proxy
        image: mcp-proxy:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: jwt-secret
              key: secret
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-proxy
spec:
  selector:
    app: mcp-proxy
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Use Cases

1. **API Gateway** - Single entry point for multiple MCP services
2. **Load Balancing** - Distribute load across multiple server instances
3. **High Availability** - Automatic failover when servers fail
4. **Security Layer** - Centralized authentication and authorization
5. **Performance** - Response caching and connection pooling
6. **Rate Limiting** - Protect backend services from overload
7. **A/B Testing** - Route traffic to different versions
8. **Monitoring** - Centralized metrics and logging
9. **Protocol Translation** - Bridge different transport types
10. **Development** - Local proxy for testing distributed systems

## Summary

FastMCP Proxy provides production-ready infrastructure for MCP services:
- **Scalability** through load balancing and connection pooling
- **Reliability** with health checks and circuit breakers
- **Security** via authentication and rate limiting
- **Performance** using caching and optimized routing
- **Observability** through metrics and logging
- **Flexibility** with middleware and custom handlers

The proxy handles the complexity of distributed MCP systems, allowing you to focus on building your services.