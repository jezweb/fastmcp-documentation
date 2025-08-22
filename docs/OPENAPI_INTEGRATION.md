# FastMCP OpenAPI & FastAPI Integration Guide

Complete guide for integrating OpenAPI/Swagger specifications and FastAPI applications with FastMCP to create MCP servers.

## Table of Contents
1. [Overview](#overview)
2. [OpenAPI Integration](#openapi-integration)
3. [FastAPI Integration](#fastapi-integration)
4. [Route Mapping](#route-mapping)
5. [Component Customization](#component-customization)
6. [Authentication](#authentication)
7. [Error Handling](#error-handling)
8. [Advanced Patterns](#advanced-patterns)
9. [Best Practices](#best-practices)

## Overview

FastMCP can automatically generate MCP servers from:
- **OpenAPI/Swagger specifications** (v3.0.x and v3.1.x)
- **FastAPI applications**
- **Any REST API with an OpenAPI spec**

This allows you to instantly make any REST API accessible to LLMs through the MCP protocol.

## OpenAPI Integration

### Basic Setup

```python
import httpx
from fastmcp import FastMCP

# Load OpenAPI specification
spec_url = "https://api.example.com/openapi.json"
spec = httpx.get(spec_url).json()

# Create authenticated HTTP client
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    timeout=30.0
)

# Generate MCP server from OpenAPI spec
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    name="API Server"
)

# All endpoints are now MCP components!
if __name__ == "__main__":
    mcp.run()
```

### Loading OpenAPI Specs

```python
# From URL
spec = httpx.get("https://api.example.com/openapi.json").json()

# From local file
import json
with open("openapi.json") as f:
    spec = json.load(f)

# From YAML file
import yaml
with open("openapi.yaml") as f:
    spec = yaml.safe_load(f)
```

### Client Configuration

```python
# Basic authentication
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    auth=("username", "password")
)

# Bearer token
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {token}"}
)

# API key in header
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"X-API-Key": API_KEY}
)

# Custom timeout and limits
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    timeout=httpx.Timeout(60.0, connect=5.0),
    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
)
```

## FastAPI Integration

### Basic FastAPI to MCP

```python
from fastapi import FastAPI
from fastmcp import FastMCP

# Existing FastAPI app
app = FastAPI()

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"id": item_id, "name": f"Item {item_id}"}

@app.post("/items")
def create_item(name: str, price: float):
    return {"name": name, "price": price}

# Convert to MCP server
mcp = FastMCP.from_fastapi(app=app)

# Run as MCP server
if __name__ == "__main__":
    mcp.run()
```

### With Authentication

```python
# FastAPI app with authentication
mcp = FastMCP.from_fastapi(
    app=app,
    httpx_client_kwargs={
        "headers": {"Authorization": f"Bearer {token}"},
        "base_url": "http://localhost:8000"
    }
)
```

### Running FastAPI App Separately

```python
# If FastAPI is running on a different server
spec = httpx.get("http://localhost:8000/openapi.json").json()

client = httpx.AsyncClient(
    base_url="http://localhost:8000",
    headers={"Authorization": f"Bearer {token}"}
)

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    name="FastAPI Server"
)
```

## Route Mapping

### Default Mapping Rules

By default, FastMCP maps OpenAPI endpoints to MCP components:
- `GET` endpoints → Resources or Resource Templates
- `POST`, `PUT`, `DELETE`, `PATCH` → Tools
- Endpoints with path parameters → Resource Templates

### Custom Route Mapping

```python
from fastmcp.server.openapi import RouteMap, MCPType

route_maps = [
    # GET with parameters → Resource Templates
    RouteMap(
        methods=["GET"],
        pattern=r".*\{.*\}.*",  # Has path parameters
        mcp_type=MCPType.RESOURCE_TEMPLATE
    ),
    
    # Static GET → Resources
    RouteMap(
        methods=["GET"],
        pattern=r"^/(?!.*\{).*",  # No parameters
        mcp_type=MCPType.RESOURCE
    ),
    
    # All POST operations → Tools
    RouteMap(
        methods=["POST"],
        mcp_type=MCPType.TOOL
    ),
    
    # Exclude internal endpoints
    RouteMap(
        pattern=r"/internal/.*",
        mcp_type=MCPType.EXCLUDE
    ),
    
    # Admin endpoints → Tools with special handling
    RouteMap(
        pattern=r"/admin/.*",
        methods=["GET", "POST", "DELETE"],
        mcp_type=MCPType.TOOL
    )
]

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    route_maps=route_maps
)
```

### Pattern Examples

```python
# Match specific paths
RouteMap(pattern=r"^/users/.*", mcp_type=MCPType.RESOURCE)

# Match by method and path
RouteMap(
    methods=["GET"],
    pattern=r"^/api/v1/.*",
    mcp_type=MCPType.RESOURCE
)

# Exclude patterns
RouteMap(pattern=r".*/debug$", mcp_type=MCPType.EXCLUDE)

# Complex patterns
RouteMap(
    pattern=r"^/api/v[0-9]+/users/\{id\}/.*",
    mcp_type=MCPType.RESOURCE_TEMPLATE
)
```

## Component Customization

### Custom Component Function

```python
def customize_component(component, route_info):
    """
    Customize generated components.
    
    Args:
        component: The generated MCP component
        route_info: HTTPRoute object with path, method, etc.
    
    Returns:
        Modified component or None to exclude
    """
    # Improve naming
    if component.name.startswith("get_"):
        component.name = component.name[4:]  # Remove 'get_' prefix
    
    # Add tags based on path
    if "/admin/" in route_info.path:
        component.tags = ["admin", "restricted"]
    elif "/public/" in route_info.path:
        component.tags = ["public"]
    
    # Enhance descriptions
    if not component.description:
        component.description = f"Access {route_info.path} via {route_info.method}"
    
    # Exclude deprecated endpoints
    if route_info.deprecated:
        return None  # Exclude this component
    
    return component

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    mcp_component_fn=customize_component
)
```

### Advanced Customization

```python
class ComponentEnhancer:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, component, route_info):
        # Complex logic based on configuration
        if route_info.path in self.config.priority_endpoints:
            component.priority = "high"
        
        # Add metadata
        component.metadata = {
            "original_path": route_info.path,
            "method": route_info.method,
            "api_version": self.extract_version(route_info.path)
        }
        
        # Transform parameters
        if hasattr(component, 'parameters'):
            for param in component.parameters:
                if param.name == 'limit':
                    param.default = 100
                    param.maximum = 1000
        
        return component
    
    def extract_version(self, path):
        # Extract API version from path
        import re
        match = re.search(r'/v(\d+)/', path)
        return match.group(1) if match else "1"

enhancer = ComponentEnhancer(config)
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    mcp_component_fn=enhancer
)
```

## Authentication

### OAuth 2.0 Integration

```python
import httpx
from httpx_oauth import OAuth2Client

# Setup OAuth client
oauth_client = OAuth2Client(
    client_id="your-client-id",
    client_secret="your-client-secret",
    token_endpoint="https://auth.example.com/token"
)

# Get token
token = await oauth_client.get_token(
    grant_type="client_credentials",
    scope="api.read api.write"
)

# Create HTTP client with token
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {token.access_token}"}
)

mcp = FastMCP.from_openapi(spec, client)
```

### API Key Authentication

```python
# Query parameter
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    params={"api_key": API_KEY}
)

# Header
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"X-API-Key": API_KEY}
)

# Custom auth class
class APIKeyAuth(httpx.Auth):
    def __init__(self, api_key):
        self.api_key = api_key
    
    def auth_flow(self, request):
        request.headers["X-API-Key"] = self.api_key
        yield request

client = httpx.AsyncClient(
    base_url="https://api.example.com",
    auth=APIKeyAuth(API_KEY)
)
```

### Token Refresh

```python
class RefreshableTokenAuth(httpx.Auth):
    def __init__(self, token_url, client_id, client_secret):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.expires_at = 0
    
    def auth_flow(self, request):
        import time
        
        # Refresh token if expired
        if time.time() >= self.expires_at:
            self.refresh_token()
        
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request
    
    def refresh_token(self):
        import time
        response = httpx.post(
            self.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
        )
        data = response.json()
        self.token = data["access_token"]
        self.expires_at = time.time() + data.get("expires_in", 3600)

auth = RefreshableTokenAuth(token_url, client_id, client_secret)
client = httpx.AsyncClient(base_url=base_url, auth=auth)
```

## Error Handling

### Response Validation

```python
def validate_response(component, route_info):
    """Add error handling to components."""
    original_handler = component.handler
    
    async def wrapped_handler(*args, **kwargs):
        try:
            result = await original_handler(*args, **kwargs)
            
            # Check for API errors in response
            if isinstance(result, dict):
                if result.get("error"):
                    return {"error": f"API Error: {result['error']}"}
                if result.get("status") == "error":
                    return {"error": result.get("message", "Unknown error")}
            
            return result
            
        except httpx.HTTPStatusError as e:
            return {
                "error": f"HTTP {e.response.status_code}",
                "message": str(e),
                "endpoint": route_info.path
            }
        except httpx.RequestError as e:
            return {
                "error": "Request failed",
                "message": str(e),
                "endpoint": route_info.path
            }
        except Exception as e:
            return {
                "error": "Unexpected error",
                "message": str(e),
                "endpoint": route_info.path
            }
    
    component.handler = wrapped_handler
    return component

mcp = FastMCP.from_openapi(
    spec,
    client,
    mcp_component_fn=validate_response
)
```

### Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

def add_retry_logic(component, route_info):
    """Add retry logic to components."""
    original_handler = component.handler
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def retrying_handler(*args, **kwargs):
        return await original_handler(*args, **kwargs)
    
    component.handler = retrying_handler
    return component
```

## Advanced Patterns

### Combining Multiple APIs

```python
# Create servers from multiple APIs
api1_mcp = FastMCP.from_openapi(api1_spec, api1_client, name="API1")
api2_mcp = FastMCP.from_openapi(api2_spec, api2_client, name="API2")

# Combine into single server
main_mcp = FastMCP("Combined API Server")
main_mcp.mount(api1_mcp, prefix="api1")
main_mcp.mount(api2_mcp, prefix="api2")

# Add custom tools
@main_mcp.tool()
async def cross_api_operation(param: str) -> dict:
    """Custom tool that uses both APIs."""
    # Use components from both APIs
    result1 = await api1_mcp.call_tool("api1_tool", {"param": param})
    result2 = await api2_mcp.call_tool("api2_tool", {"data": result1})
    return {"combined": result2}
```

### Caching Responses

```python
from functools import lru_cache
import hashlib
import json

class CachedComponentWrapper:
    def __init__(self, cache_ttl=300):
        self.cache = {}
        self.cache_times = {}
        self.ttl = cache_ttl
    
    def __call__(self, component, route_info):
        # Only cache GET requests
        if route_info.method != "GET":
            return component
        
        original_handler = component.handler
        
        async def cached_handler(*args, **kwargs):
            import time
            
            # Create cache key
            cache_key = hashlib.md5(
                json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True).encode()
            ).hexdigest()
            
            # Check cache
            if cache_key in self.cache:
                if time.time() - self.cache_times[cache_key] < self.ttl:
                    return self.cache[cache_key]
            
            # Call original handler
            result = await original_handler(*args, **kwargs)
            
            # Store in cache
            self.cache[cache_key] = result
            self.cache_times[cache_key] = time.time()
            
            return result
        
        component.handler = cached_handler
        return component

wrapper = CachedComponentWrapper(cache_ttl=600)
mcp = FastMCP.from_openapi(spec, client, mcp_component_fn=wrapper)
```

### Rate Limiting

```python
import asyncio
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests=100, window=60):
        self.max_requests = max_requests
        self.window = window
        self.requests = defaultdict(list)
    
    async def check_rate_limit(self, key="global"):
        now = time.time()
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < self.window
        ]
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            wait_time = self.window - (now - self.requests[key][0])
            await asyncio.sleep(wait_time)
        
        # Record request
        self.requests[key].append(now)

rate_limiter = RateLimiter(max_requests=100, window=60)

def add_rate_limiting(component, route_info):
    original_handler = component.handler
    
    async def rate_limited_handler(*args, **kwargs):
        await rate_limiter.check_rate_limit(route_info.path)
        return await original_handler(*args, **kwargs)
    
    component.handler = rate_limited_handler
    return component

mcp = FastMCP.from_openapi(
    spec,
    client,
    mcp_component_fn=add_rate_limiting
)
```

## Best Practices

### 1. Specification Validation

```python
def validate_openapi_spec(spec):
    """Validate OpenAPI specification."""
    required_fields = ["openapi", "info", "paths"]
    
    for field in required_fields:
        if field not in spec:
            raise ValueError(f"Invalid OpenAPI spec: missing {field}")
    
    # Check version
    version = spec["openapi"]
    if not version.startswith("3."):
        raise ValueError(f"Unsupported OpenAPI version: {version}")
    
    return True

# Validate before using
validate_openapi_spec(spec)
mcp = FastMCP.from_openapi(spec, client)
```

### 2. Environment-Specific Configuration

```python
import os
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

def get_api_config(env: Environment):
    configs = {
        Environment.DEVELOPMENT: {
            "base_url": "http://localhost:8000",
            "timeout": 60.0,
            "verify_ssl": False
        },
        Environment.STAGING: {
            "base_url": "https://staging-api.example.com",
            "timeout": 30.0,
            "verify_ssl": True
        },
        Environment.PRODUCTION: {
            "base_url": "https://api.example.com",
            "timeout": 30.0,
            "verify_ssl": True
        }
    }
    return configs[env]

# Use environment-specific config
env = Environment(os.getenv("ENVIRONMENT", "development"))
config = get_api_config(env)

client = httpx.AsyncClient(
    base_url=config["base_url"],
    timeout=config["timeout"],
    verify=config["verify_ssl"]
)
```

### 3. Monitoring and Logging

```python
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def add_logging(component, route_info):
    """Add logging to all components."""
    original_handler = component.handler
    
    async def logged_handler(*args, **kwargs):
        start_time = datetime.now()
        request_id = f"{route_info.path}_{start_time.timestamp()}"
        
        logger.info(f"[{request_id}] Starting {route_info.method} {route_info.path}")
        
        try:
            result = await original_handler(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"[{request_id}] Completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"[{request_id}] Failed after {duration:.2f}s: {e}")
            raise
    
    component.handler = logged_handler
    return component
```

### 4. Documentation Enhancement

```python
def enhance_documentation(component, route_info):
    """Enhance component documentation from OpenAPI spec."""
    
    # Add parameter descriptions
    if hasattr(route_info, 'parameters'):
        param_docs = []
        for param in route_info.parameters:
            param_docs.append(f"- {param.name}: {param.description}")
        
        if param_docs:
            component.description += "\n\nParameters:\n" + "\n".join(param_docs)
    
    # Add response information
    if hasattr(route_info, 'responses'):
        response_docs = []
        for code, response in route_info.responses.items():
            response_docs.append(f"- {code}: {response.description}")
        
        if response_docs:
            component.description += "\n\nResponses:\n" + "\n".join(response_docs)
    
    # Add examples if available
    if hasattr(route_info, 'examples'):
        component.description += f"\n\nExample: {route_info.examples[0]}"
    
    return component
```

### 5. Testing Generated Servers

```python
import pytest
from fastmcp import Client

@pytest.fixture
async def mcp_client():
    """Fixture for testing MCP server."""
    spec = load_openapi_spec()
    http_client = create_test_client()
    mcp = FastMCP.from_openapi(spec, http_client)
    
    async with Client(mcp) as client:
        yield client

async def test_generated_tools(mcp_client):
    """Test that tools are generated correctly."""
    tools = await mcp_client.list_tools()
    
    # Check expected tools exist
    tool_names = [tool.name for tool in tools]
    assert "create_user" in tool_names
    assert "get_user" in tool_names

async def test_generated_resources(mcp_client):
    """Test that resources are generated correctly."""
    resources = await mcp_client.list_resources()
    
    # Check expected resources exist
    resource_uris = [r.uri for r in resources]
    assert "api://users" in resource_uris

async def test_tool_execution(mcp_client):
    """Test executing a generated tool."""
    result = await mcp_client.call_tool(
        "create_user",
        {"name": "Test User", "email": "test@example.com"}
    )
    
    assert result.get("id") is not None
    assert result.get("name") == "Test User"
```

## Common Issues and Solutions

### Issue: Large OpenAPI Specs

```python
# Filter spec to only needed paths
def filter_openapi_spec(spec, included_paths):
    """Filter OpenAPI spec to only include specific paths."""
    filtered_spec = spec.copy()
    filtered_spec["paths"] = {
        path: operations
        for path, operations in spec["paths"].items()
        if any(pattern in path for pattern in included_paths)
    }
    return filtered_spec

# Use filtered spec
filtered = filter_openapi_spec(spec, ["/users", "/products"])
mcp = FastMCP.from_openapi(filtered, client)
```

### Issue: Parameter Conflicts

```python
def resolve_parameter_conflicts(component, route_info):
    """Resolve naming conflicts in parameters."""
    if hasattr(component, 'parameters'):
        seen = set()
        for param in component.parameters:
            original = param.name
            counter = 1
            while param.name in seen:
                param.name = f"{original}_{counter}"
                counter += 1
            seen.add(param.name)
    
    return component
```

### Issue: Slow API Responses

```python
# Add timeout handling
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    timeout=httpx.Timeout(
        timeout=30.0,  # Total timeout
        connect=5.0,   # Connection timeout
        read=25.0,     # Read timeout
        write=5.0      # Write timeout
    )
)

# Add concurrent request limiting
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0
    )
)
```

## Summary

FastMCP's OpenAPI and FastAPI integration provides:
- **Instant API wrapping** - Any REST API becomes MCP-compatible
- **Automatic component generation** - Tools and resources from endpoints
- **Flexible customization** - Route mapping and component transformation
- **Production-ready features** - Authentication, caching, rate limiting
- **Easy testing** - Generated servers work with FastMCP Client

This makes it trivial to expose existing APIs to LLMs through the MCP protocol, enabling AI agents to interact with any REST API seamlessly.