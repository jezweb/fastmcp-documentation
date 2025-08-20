# FastMCP Common Patterns and Solutions

This document captures proven patterns and solutions for FastMCP development based on real-world implementations.

## Table of Contents
1. [Server Initialization Patterns](#server-initialization-patterns)
2. [Tool Implementation Patterns](#tool-implementation-patterns)
3. [Resource Management Patterns](#resource-management-patterns)
4. [API Integration Patterns](#api-integration-patterns)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Caching Patterns](#caching-patterns)
7. [Authentication Patterns](#authentication-patterns)
8. [Testing Patterns](#testing-patterns)
9. [Deployment Patterns](#deployment-patterns)
10. [Performance Patterns](#performance-patterns)

## Server Initialization Patterns

### Pattern 1: Simple Module-Level Server
**Use Case**: Small, single-purpose servers
```python
from fastmcp import FastMCP

# MUST be at module level for FastMCP Cloud
mcp = FastMCP(
    name="simple-server",
    version="1.0.0"
)

@mcp.tool()
async def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"
```

### Pattern 2: Factory Function with Module Export
**Use Case**: Complex servers needing initialization logic
```python
from fastmcp import FastMCP
import os

def create_server() -> FastMCP:
    """Factory function for server creation."""
    server = FastMCP(
        name="complex-server",
        version="1.0.0"
    )
    
    # Complex setup logic
    if os.getenv("ENABLE_ADVANCED"):
        setup_advanced_features(server)
    
    return server

def setup_advanced_features(server: FastMCP):
    @server.tool()
    async def advanced_tool():
        pass

# CRITICAL: Must export at module level
mcp = create_server()

if __name__ == "__main__":
    mcp.run()
```

### Pattern 3: Class-Based Organization with Module Export
**Use Case**: Large servers with grouped functionality
```python
from fastmcp import FastMCP

class ServerBuilder:
    def __init__(self):
        self.mcp = FastMCP(name="organized-server")
        self.setup_tools()
        self.setup_resources()
    
    def setup_tools(self):
        @self.mcp.tool()
        async def tool_one():
            pass
    
    def setup_resources(self):
        @self.mcp.resource("data://config")
        async def get_config():
            pass

# CRITICAL: Export instance at module level
builder = ServerBuilder()
mcp = builder.mcp
```

## Tool Implementation Patterns

### Pattern 1: Simple Synchronous Tool
```python
@mcp.tool()
def calculate(operation: str, a: float, b: float) -> dict:
    """Perform calculation with validation."""
    if operation not in ["add", "subtract", "multiply", "divide"]:
        return {"error": "Invalid operation"}
    
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None
    }
    
    result = operations[operation](a, b)
    if result is None:
        return {"error": "Division by zero"}
    
    return {"result": result}
```

### Pattern 2: Async Tool with External API
```python
import httpx

@mcp.tool()
async def fetch_data(endpoint: str, params: dict = None) -> dict:
    """Fetch data from external API with error handling."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/{endpoint}",
                params=params,
                timeout=30.0
            )
            response.raise_for_status()
            return {"success": True, "data": response.json()}
    except httpx.TimeoutException:
        return {"error": "Request timeout"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
```

### Pattern 3: Tool with Complex Input Validation
```python
from pydantic import BaseModel, Field, validator

class SearchParams(BaseModel):
    query: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filters: dict = Field(default_factory=dict)
    
    @validator("query")
    def clean_query(cls, v):
        return v.strip()

@mcp.tool()
async def search(params: SearchParams) -> dict:
    """Search with validated parameters."""
    # Pydantic handles validation automatically
    results = await perform_search(
        params.query,
        params.limit,
        params.offset,
        params.filters
    )
    return {"results": results, "total": len(results)}
```

## Resource Management Patterns

### Pattern 1: Static Resource
```python
@mcp.resource("info://server")
def server_info() -> dict:
    """Provide server metadata."""
    return {
        "name": "My Server",
        "version": "1.0.0",
        "capabilities": ["search", "process", "analyze"]
    }
```

### Pattern 2: Dynamic Resource with Caching
```python
import time
from functools import lru_cache

@mcp.resource("data://statistics")
@lru_cache(maxsize=1)
def get_statistics() -> dict:
    """Get cached statistics (refreshes every call)."""
    # Clear cache periodically
    get_statistics.cache_clear()
    
    return {
        "timestamp": time.time(),
        "active_connections": count_connections(),
        "processed_requests": get_request_count()
    }
```

### Pattern 3: Resource Template with Parameters
```python
@mcp.resource("user://{user_id}/profile")
async def get_user_profile(user_id: str) -> dict:
    """Get user profile by ID."""
    try:
        user = await fetch_user_from_db(user_id)
        return {
            "id": user_id,
            "name": user.name,
            "email": user.email,
            "created": user.created_at.isoformat()
        }
    except UserNotFound:
        return {"error": f"User {user_id} not found"}
```

## API Integration Patterns

### Pattern 1: Manual API Integration with Client Management
```python
import httpx
from typing import Optional

class APIClient:
    _instance: Optional[httpx.AsyncClient] = None
    
    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        if cls._instance is None:
            cls._instance = httpx.AsyncClient(
                base_url=os.getenv("API_BASE_URL"),
                headers={
                    "Authorization": f"Bearer {os.getenv('API_KEY')}",
                    "Content-Type": "application/json"
                },
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=5)
            )
        return cls._instance
    
    @classmethod
    async def cleanup(cls):
        if cls._instance:
            await cls._instance.aclose()
            cls._instance = None

@mcp.tool()
async def api_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make API request with managed client."""
    client = await APIClient.get_client()
    
    try:
        response = await client.request(method, endpoint, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}
```

### Pattern 2: OpenAPI Auto-Generation with Customization
```python
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

# Load OpenAPI spec
spec = load_openapi_spec("https://api.example.com/openapi.json")

# Create MCP with custom route mapping
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=httpx.AsyncClient(
        base_url="https://api.example.com",
        headers={"Authorization": f"Bearer {API_KEY}"}
    ),
    route_maps=[
        # GET endpoints with parameters become resource templates
        RouteMap(
            methods=["GET"],
            pattern=r"/users/\{user_id\}",
            mcp_type=MCPType.RESOURCE_TEMPLATE
        ),
        # POST endpoints become tools
        RouteMap(
            methods=["POST"],
            mcp_type=MCPType.TOOL
        ),
        # Specific endpoints can be excluded
        RouteMap(
            pattern=r"/internal/.*",
            mcp_type=MCPType.EXCLUDE
        )
    ]
)

# Add custom tool on top of generated ones
@mcp.tool()
async def process_api_response(data: dict) -> dict:
    """Custom processing of API responses."""
    return transform_data(data)
```

### Pattern 3: Rate-Limited API Integration
```python
import asyncio
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window  # seconds
        self.requests = []
    
    async def acquire(self):
        now = datetime.now()
        # Remove old requests outside time window
        self.requests = [
            req for req in self.requests 
            if now - req < timedelta(seconds=self.time_window)
        ]
        
        if len(self.requests) >= self.max_requests:
            sleep_time = (self.requests[0] + timedelta(seconds=self.time_window) - now).total_seconds()
            await asyncio.sleep(sleep_time)
            return await self.acquire()
        
        self.requests.append(now)

# Create rate limiter (e.g., 100 requests per minute)
rate_limiter = RateLimiter(100, 60)

@mcp.tool()
async def rate_limited_api_call(endpoint: str) -> dict:
    """API call with rate limiting."""
    await rate_limiter.acquire()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/{endpoint}")
        return response.json()
```

## Error Handling Patterns

### Pattern 1: Structured Error Responses
```python
from enum import Enum

class ErrorCode(Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    API_ERROR = "API_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMITED = "RATE_LIMITED"

def create_error(code: ErrorCode, message: str, details: dict = None) -> dict:
    """Create structured error response."""
    return {
        "error": True,
        "code": code.value,
        "message": message,
        "details": details or {}
    }

@mcp.tool()
async def safe_operation(param: str) -> dict:
    """Operation with structured error handling."""
    if not param:
        return create_error(
            ErrorCode.VALIDATION_ERROR,
            "Parameter is required",
            {"field": "param"}
        )
    
    try:
        result = await perform_operation(param)
        return {"success": True, "data": result}
    except NotFoundException as e:
        return create_error(ErrorCode.NOT_FOUND, str(e))
    except APIException as e:
        return create_error(ErrorCode.API_ERROR, str(e), {"status": e.status})
```

### Pattern 2: Retry with Exponential Backoff
```python
import asyncio
from typing import TypeVar, Callable

T = TypeVar('T')

async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
) -> T:
    """Retry function with exponential backoff."""
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                await asyncio.sleep(min(delay, max_delay))
                delay *= exponential_base
    
    raise last_exception

@mcp.tool()
async def resilient_api_call(endpoint: str) -> dict:
    """API call with automatic retry."""
    async def make_call():
        async with httpx.AsyncClient() as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.json()
    
    try:
        data = await retry_with_backoff(make_call)
        return {"success": True, "data": data}
    except Exception as e:
        return {"error": f"Failed after retries: {e}"}
```

## Caching Patterns

### Pattern 1: Time-Based Cache
```python
import time
from typing import Any, Optional

class TimeBasedCache:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
            else:
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def invalidate(self, pattern: str = None):
        if pattern:
            keys_to_delete = [k for k in self.cache if pattern in k]
            for key in keys_to_delete:
                del self.cache[key]
                del self.timestamps[key]
        else:
            self.cache.clear()
            self.timestamps.clear()

# Global cache instance
cache = TimeBasedCache(ttl=300)

@mcp.tool()
async def cached_fetch(resource_id: str) -> dict:
    """Fetch with caching."""
    cache_key = f"resource:{resource_id}"
    
    # Check cache
    cached_data = cache.get(cache_key)
    if cached_data:
        return {"data": cached_data, "from_cache": True}
    
    # Fetch fresh data
    data = await fetch_from_api(resource_id)
    cache.set(cache_key, data)
    
    return {"data": data, "from_cache": False}
```

### Pattern 2: LRU Cache with Async
```python
from functools import lru_cache
import hashlib
import json

def make_cache_key(*args, **kwargs) -> str:
    """Create cache key from arguments."""
    key_data = {"args": args, "kwargs": kwargs}
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode()).hexdigest()

# Store async results in sync cache
@lru_cache(maxsize=100)
def _cached_result(cache_key: str):
    return None

async def cached_async_operation(param: str) -> dict:
    """Async operation with caching."""
    cache_key = make_cache_key(param)
    
    # Check cache
    result = _cached_result(cache_key)
    if result is not None:
        return result
    
    # Compute result
    result = await expensive_async_operation(param)
    
    # Store in cache (hack: replace cached None)
    _cached_result.cache_clear()
    _cached_result(cache_key)
    _cached_result.__wrapped__(cache_key)  # Direct cache update
    
    return result
```

## Authentication Patterns

### Pattern 1: API Key Authentication
```python
import os
from typing import Optional

class AuthConfig:
    API_KEY: Optional[str] = os.getenv("API_KEY")
    API_SECRET: Optional[str] = os.getenv("API_SECRET")
    
    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("API_KEY environment variable is required")
        return True

# Validate on module load
AuthConfig.validate()

@mcp.tool()
async def authenticated_request(endpoint: str) -> dict:
    """Make authenticated API request."""
    headers = {
        "X-API-Key": AuthConfig.API_KEY,
        "X-API-Secret": AuthConfig.API_SECRET
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/{endpoint}",
            headers=headers
        )
        return response.json()
```

### Pattern 2: OAuth2 with Token Refresh
```python
import asyncio
from datetime import datetime, timedelta

class OAuth2Client:
    def __init__(self):
        self.access_token = None
        self.refresh_token = None
        self.expires_at = None
        self.lock = asyncio.Lock()
    
    async def get_token(self) -> str:
        async with self.lock:
            if self.expires_at and datetime.now() < self.expires_at:
                return self.access_token
            
            # Refresh token
            await self.refresh_access_token()
            return self.access_token
    
    async def refresh_access_token(self):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://auth.example.com/token",
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self.refresh_token or "initial_auth",
                    "client_id": os.getenv("CLIENT_ID"),
                    "client_secret": os.getenv("CLIENT_SECRET")
                }
            )
            token_data = response.json()
            
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token", self.refresh_token)
            expires_in = token_data.get("expires_in", 3600)
            self.expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

oauth_client = OAuth2Client()

@mcp.tool()
async def oauth_protected_request(endpoint: str) -> dict:
    """Request with OAuth2 authentication."""
    token = await oauth_client.get_token()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{BASE_URL}/{endpoint}",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
```

## Testing Patterns

### Pattern 1: Unit Testing Tools
```python
import pytest
from fastmcp import FastMCP
from fastmcp.testing import create_test_client

@pytest.fixture
def test_server():
    """Create test server instance."""
    mcp = FastMCP("test-server")
    
    @mcp.tool()
    async def test_tool(param: str) -> str:
        return f"Result: {param}"
    
    return mcp

@pytest.mark.asyncio
async def test_tool_execution(test_server):
    """Test tool execution."""
    async with create_test_client(test_server) as client:
        result = await client.call_tool("test_tool", {"param": "test"})
        assert result.data == "Result: test"
```

### Pattern 2: Integration Testing with Mocks
```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_api_integration():
    """Test API integration with mocked responses."""
    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value = AsyncMock(
            status_code=200,
            json=lambda: {"data": "mocked"}
        )
        
        result = await api_tool("test-endpoint")
        assert result["data"] == "mocked"
        mock_get.assert_called_once()
```

### Pattern 3: End-to-End Testing
```python
import subprocess
import time
import httpx

def test_server_deployment():
    """Test full server deployment."""
    # Start server
    process = subprocess.Popen(
        ["python", "server.py", "--transport", "http", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for startup
    time.sleep(2)
    
    try:
        # Test server is responding
        response = httpx.get("http://localhost:8000/health")
        assert response.status_code == 200
        
        # Test MCP endpoint
        response = httpx.post(
            "http://localhost:8000/mcp",
            json={"method": "tools/list"}
        )
        assert "result" in response.json()
    finally:
        process.terminate()
        process.wait()
```

## Deployment Patterns

### Pattern 1: Environment-Based Configuration
```python
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class Config:
    ENV = Environment(os.getenv("ENVIRONMENT", "development"))
    
    # Environment-specific settings
    SETTINGS = {
        Environment.DEVELOPMENT: {
            "debug": True,
            "cache_ttl": 60,
            "log_level": "DEBUG"
        },
        Environment.STAGING: {
            "debug": True,
            "cache_ttl": 300,
            "log_level": "INFO"
        },
        Environment.PRODUCTION: {
            "debug": False,
            "cache_ttl": 3600,
            "log_level": "WARNING"
        }
    }
    
    @classmethod
    def get(cls, key: str):
        return cls.SETTINGS[cls.ENV].get(key)

# Use configuration
mcp = FastMCP(
    name="adaptive-server",
    debug=Config.get("debug")
)
```

### Pattern 2: Multi-Stage Docker Deployment
```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

CMD ["python", "server.py"]
```

### Pattern 3: Health Check Implementation
```python
from datetime import datetime
import psutil
import asyncio

@mcp.resource("health://status")
async def health_check() -> dict:
    """Comprehensive health check."""
    checks = {}
    
    # Check API connectivity
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=5)
            checks["api"] = response.status_code == 200
    except:
        checks["api"] = False
    
    # Check database
    try:
        checks["database"] = await check_db_connection()
    except:
        checks["database"] = False
    
    # System resources
    checks["memory_percent"] = psutil.virtual_memory().percent
    checks["cpu_percent"] = psutil.cpu_percent(interval=1)
    
    # Overall status
    all_healthy = (
        checks.get("api", False) and 
        checks.get("database", False) and
        checks["memory_percent"] < 90 and
        checks["cpu_percent"] < 90
    )
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "version": "1.0.0"
    }
```

## Performance Patterns

### Pattern 1: Parallel Processing
```python
import asyncio
from typing import List, Dict

@mcp.tool()
async def batch_process(items: List[str]) -> Dict[str, any]:
    """Process multiple items in parallel."""
    async def process_single(item: str):
        try:
            result = await process_item(item)
            return {"item": item, "result": result, "success": True}
        except Exception as e:
            return {"item": item, "error": str(e), "success": False}
    
    # Process all items in parallel
    tasks = [process_single(item) for item in items]
    results = await asyncio.gather(*tasks)
    
    # Aggregate results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    return {
        "total": len(items),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }
```

### Pattern 2: Connection Pooling
```python
from contextlib import asynccontextmanager
import asyncpg

class DatabasePool:
    _pool = None
    
    @classmethod
    async def get_pool(cls):
        if cls._pool is None:
            cls._pool = await asyncpg.create_pool(
                dsn=os.getenv("DATABASE_URL"),
                min_size=2,
                max_size=10,
                max_queries=50000,
                max_inactive_connection_lifetime=300
            )
        return cls._pool
    
    @classmethod
    @asynccontextmanager
    async def get_connection(cls):
        pool = await cls.get_pool()
        async with pool.acquire() as connection:
            yield connection
    
    @classmethod
    async def cleanup(cls):
        if cls._pool:
            await cls._pool.close()

@mcp.tool()
async def database_query(query: str, params: list = None) -> dict:
    """Execute database query with connection pooling."""
    async with DatabasePool.get_connection() as conn:
        try:
            results = await conn.fetch(query, *(params or []))
            return {"data": [dict(r) for r in results]}
        except Exception as e:
            return {"error": str(e)}
```

### Pattern 3: Response Streaming
```python
from typing import AsyncIterator
import json

@mcp.tool()
async def stream_large_dataset(query: str) -> AsyncIterator[str]:
    """Stream large dataset in chunks."""
    async with DatabasePool.get_connection() as conn:
        async with conn.transaction():
            cursor = await conn.cursor(query)
            
            while True:
                rows = await cursor.fetch(100)  # Fetch 100 rows at a time
                if not rows:
                    break
                
                for row in rows:
                    yield json.dumps(dict(row)) + "\n"
```

## Common Anti-Patterns to Avoid

### Anti-Pattern 1: Function-Wrapped Server (Breaks FastMCP Cloud)
```python
# ❌ WRONG - Server not accessible at module level
def create_server():
    mcp = FastMCP("my-server")
    return mcp

if __name__ == "__main__":
    server = create_server()  # Too late for FastMCP Cloud!
    server.run()
```

### Anti-Pattern 2: Blocking Operations in Async Context
```python
# ❌ WRONG - Blocks event loop
@mcp.tool()
async def bad_async_tool():
    time.sleep(5)  # Blocks entire event loop!
    return "done"

# ✅ CORRECT - Use async sleep
@mcp.tool()
async def good_async_tool():
    await asyncio.sleep(5)
    return "done"
```

### Anti-Pattern 3: Global Mutable State
```python
# ❌ WRONG - Shared mutable state causes issues
results = []

@mcp.tool()
async def add_result(data: str):
    results.append(data)  # Race conditions in async context!
    return {"count": len(results)}

# ✅ CORRECT - Use proper state management
import asyncio

class StateManager:
    def __init__(self):
        self.results = []
        self.lock = asyncio.Lock()
    
    async def add_result(self, data: str):
        async with self.lock:
            self.results.append(data)
            return len(self.results)

state = StateManager()

@mcp.tool()
async def add_result(data: str):
    count = await state.add_result(data)
    return {"count": count}
```

## Resource Template Patterns

### Pattern 1: Single Parameter Template
```python
@mcp.resource("config://{environment}")
def get_config(environment: str) -> dict:
    """Get configuration by environment."""
    configs = {
        "dev": {"debug": True, "api_url": "http://localhost:8000"},
        "prod": {"debug": False, "api_url": "https://api.example.com"}
    }
    return configs.get(environment, {})
```

### Pattern 2: Multi-Parameter Template
```python
@mcp.resource("metrics://{service}/{metric}/{period}")
async def get_metrics(service: str, metric: str, period: str) -> dict:
    """Get service metrics for period."""
    data = await fetch_metrics(service, metric, period)
    return {
        "service": service,
        "metric": metric,
        "period": period,
        "values": data
    }
```

### Pattern 3: Nested Resource Templates
```python
@mcp.resource("org://{org_id}/team/{team_id}/members")
async def get_team_members(org_id: str, team_id: str) -> list:
    """Get team members with org context."""
    return await db.query(
        "SELECT * FROM members WHERE org_id = ? AND team_id = ?",
        [org_id, team_id]
    )
```

## Elicitation Handler Patterns

### Pattern 1: Type-Based Elicitation
```python
from dataclasses import dataclass
from typing import Union

@dataclass
class Confirmation:
    confirmed: bool
    reason: str = ""

async def elicitation_handler(message: str, response_type: type, context: dict):
    """Handle different elicitation types."""
    
    if response_type == str:
        return input(f"{message}: ")
    
    elif response_type == bool:
        response = input(f"{message} (y/n): ")
        return response.lower() == 'y'
    
    elif response_type == Confirmation:
        print(f"{message}")
        confirmed = input("Confirm (y/n): ").lower() == 'y'
        if not confirmed:
            reason = input("Reason for declining: ")
        return Confirmation(confirmed, reason)
```

### Pattern 2: Context-Aware Elicitation
```python
async def smart_elicitation_handler(message: str, response_type: type, context: dict):
    """Use context to provide better UX."""
    
    # Check if we can auto-fill from context
    if "default_value" in context:
        prompt = f"{message} [{context['default_value']}]: "
        response = input(prompt)
        return response or context['default_value']
    
    # Provide choices if available
    if "choices" in context:
        print(f"{message}")
        for i, choice in enumerate(context['choices'], 1):
            print(f"  {i}. {choice}")
        selection = int(input("Select: ")) - 1
        return context['choices'][selection]
    
    # Default input
    return input(f"{message}: ")
```

## Progress Tracking Patterns

### Pattern 1: Batch Processing with Progress
```python
@mcp.tool()
async def batch_import(file_path: str) -> dict:
    """Import data with detailed progress."""
    
    # Phase 1: Reading file
    await report_progress(0, 3, "Reading file...")
    data = await read_file(file_path)
    
    # Phase 2: Validating
    await report_progress(1, 3, "Validating data...")
    valid_items = []
    for i, item in enumerate(data):
        await report_progress(
            progress=i,
            total=len(data),
            message=f"Validating item {i+1}/{len(data)}"
        )
        if validate(item):
            valid_items.append(item)
    
    # Phase 3: Importing
    await report_progress(2, 3, "Importing to database...")
    imported = await import_to_db(valid_items)
    
    await report_progress(3, 3, "Complete!")
    return {"imported": len(imported), "skipped": len(data) - len(imported)}
```

### Pattern 2: Indeterminate Progress
```python
@mcp.tool()
async def analyze_data(data: dict) -> dict:
    """Long analysis with indeterminate progress."""
    
    stages = [
        "Initializing analysis engine",
        "Loading reference data",
        "Running statistical analysis",
        "Generating insights",
        "Preparing report"
    ]
    
    for stage in stages:
        # No total - shows as spinner/indeterminate
        await report_progress(
            progress=stages.index(stage),
            total=None,
            message=stage
        )
        await perform_stage(stage, data)
    
    return {"status": "complete"}
```

## Sampling Integration Patterns

### Pattern 1: Content Enhancement
```python
@mcp.tool()
async def enhance_description(text: str) -> str:
    """Enhance text using LLM."""
    
    enhanced = await request_sampling(
        messages=[{
            "role": "system",
            "content": "You are a professional copywriter. Enhance the following text while maintaining its core message."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.7,
        max_tokens=500
    )
    
    return enhanced
```

### Pattern 2: Intelligent Decision Making
```python
@mcp.tool()
async def classify_ticket(ticket: dict) -> dict:
    """Classify support ticket using AI."""
    
    classification_prompt = f"""
    Classify this support ticket:
    Subject: {ticket['subject']}
    Description: {ticket['description']}
    
    Categories: Bug, Feature Request, Question, Complaint
    Priority: Low, Medium, High, Critical
    
    Return as JSON: {{"category": "...", "priority": "...", "reason": "..."}}
    """
    
    result = await request_sampling(
        messages=[{"role": "user", "content": classification_prompt}],
        temperature=0.3,  # Lower for consistency
        response_format="json"
    )
    
    return json.loads(result)
```

## OpenAPI/Tool Transformation Patterns

### Pattern 1: Simplifying Complex APIs
```python
# Original complex API tool
complex_api = mcp.get_tool("complex_api_operation")

# Create simplified version
simple_api = Tool.from_tool(
    complex_api,
    name="simple_operation",
    description="Simplified API operation",
    transform_args={
        # Hide complex authentication
        "auth_token": ArgTransform(hide=True, default=get_token()),
        "api_version": ArgTransform(hide=True, default="v2"),
        
        # Simplify parameters
        "options": ArgTransform(
            description="Simple options (basic/advanced)",
            default="basic"
        )
    }
)

mcp.add_tool(simple_api)
```

### Pattern 2: Creating Workflow Tools
```python
def create_workflow_tool(steps: list):
    """Create a tool that executes multiple steps."""
    
    async def execute_workflow(**kwargs) -> dict:
        results = {}
        total_steps = len(steps)
        
        for i, step in enumerate(steps):
            await report_progress(i, total_steps, f"Executing: {step.name}")
            
            # Transform kwargs for this step
            step_kwargs = step.transform_kwargs(kwargs)
            
            # Execute step
            result = await step.execute(**step_kwargs)
            results[step.name] = result
            
            # Check if should continue
            if not step.should_continue(result):
                break
        
        return results
    
    return Tool(
        name="workflow",
        description="Execute multi-step workflow",
        fn=execute_workflow
    )
```

## Summary

These patterns represent proven solutions to common FastMCP development challenges. Key principles:

1. **Always export server at module level** for FastMCP Cloud
2. **Use async/await properly** - don't block the event loop
3. **Handle errors gracefully** with structured responses
4. **Cache appropriately** to reduce load
5. **Manage connections efficiently** with pooling
6. **Test thoroughly** at multiple levels
7. **Configure flexibly** with environment variables
8. **Monitor health** with comprehensive checks
9. **Use resource templates** for dynamic content generation
10. **Implement elicitation** for interactive workflows
11. **Add progress tracking** for better UX
12. **Leverage sampling** for AI-powered features
13. **Transform tools** to simplify complex interfaces

Following these patterns will help you build robust, scalable, and maintainable FastMCP servers.