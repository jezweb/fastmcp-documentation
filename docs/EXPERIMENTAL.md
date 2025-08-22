# FastMCP Experimental Features

Documentation for experimental and advanced features in FastMCP that are still evolving or require special configuration.

> ⚠️ **Note**: Features documented here are experimental and may change in future versions. Use with caution in production environments.

## Table of Contents
1. [Server Composition](#server-composition)
2. [Middleware System](#middleware-system)
3. [Advanced Context Management](#advanced-context-management)
4. [WebSocket Transport](#websocket-transport)
5. [Streaming Responses](#streaming-responses)
6. [Plugin System](#plugin-system)
7. [Distributed Tracing](#distributed-tracing)
8. [GraphQL Integration](#graphql-integration)

## Server Composition

Server composition allows mounting multiple MCP servers as sub-servers, creating modular and reusable components.

### Basic Composition

```python
from fastmcp import FastMCP

# Create sub-servers
auth_server = FastMCP("Auth Service")
data_server = FastMCP("Data Service")
analytics_server = FastMCP("Analytics Service")

# Define tools in sub-servers
@auth_server.tool()
async def authenticate(username: str, password: str) -> dict:
    """Authenticate user."""
    return {"token": "jwt-token", "user": username}

@data_server.tool()
async def fetch_data(query: str) -> list:
    """Fetch data based on query."""
    return [{"id": 1, "data": query}]

# Main server
main_server = FastMCP("Main Application")

# Mount sub-servers with prefixes
main_server.mount(auth_server, prefix="auth")
main_server.mount(data_server, prefix="data")
main_server.mount(analytics_server, prefix="analytics")

# Tools are now available as:
# - auth_authenticate
# - data_fetch_data
# - analytics_*
```

### Dynamic Server Composition

```python
class DynamicServerManager:
    def __init__(self):
        self.main_server = FastMCP("Dynamic Server")
        self.mounted_servers = {}
    
    def add_service(self, name: str, service: FastMCP, prefix: str = None):
        """Dynamically add a service."""
        prefix = prefix or name.lower()
        self.main_server.mount(service, prefix=prefix)
        self.mounted_servers[name] = service
    
    def remove_service(self, name: str):
        """Remove a mounted service."""
        if name in self.mounted_servers:
            # Unmount logic (experimental)
            self.main_server.unmount(self.mounted_servers[name])
            del self.mounted_servers[name]

# Usage
manager = DynamicServerManager()

# Add services dynamically
if enable_premium_features:
    premium_server = FastMCP("Premium Features")
    manager.add_service("premium", premium_server)

if enable_admin:
    admin_server = FastMCP("Admin Tools")
    manager.add_service("admin", admin_server, prefix="admin")
```

### Cross-Server Communication

```python
# Sub-servers can communicate
@data_server.tool()
async def authenticated_fetch(token: str, query: str) -> dict:
    """Fetch data with authentication."""
    # Call auth server's tool
    auth_result = await auth_server.call_tool("validate_token", {"token": token})
    
    if auth_result["valid"]:
        data = await fetch_data(query)
        return {"data": data, "user": auth_result["user"]}
    else:
        return {"error": "Invalid token"}
```

## Middleware System

FastMCP's middleware system allows intercepting and modifying requests/responses at various stages.

### Request/Response Middleware

```python
from fastmcp import FastMCP, Middleware

mcp = FastMCP("Server with Middleware")

@mcp.middleware()
async def logging_middleware(request, next):
    """Log all requests and responses."""
    import time
    start = time.time()
    
    print(f"[{request.id}] {request.method} {request.path}")
    
    # Call next middleware or handler
    response = await next(request)
    
    duration = time.time() - start
    print(f"[{request.id}] Response: {response.status} in {duration:.2f}s")
    
    return response

@mcp.middleware()
async def auth_middleware(request, next):
    """Check authentication."""
    if request.path.startswith("/admin/"):
        token = request.headers.get("Authorization")
        if not token or not validate_token(token):
            return ErrorResponse("Unauthorized", 401)
    
    return await next(request)
```

### Middleware Order and Priority

```python
# Middleware with priority
@mcp.middleware(priority=10)  # Higher priority runs first
async def first_middleware(request, next):
    request.context["step"] = 1
    return await next(request)

@mcp.middleware(priority=5)
async def second_middleware(request, next):
    request.context["step"] = 2
    return await next(request)

# Global error handling middleware
@mcp.middleware(priority=100)
async def error_handler_middleware(request, next):
    try:
        return await next(request)
    except ValidationError as e:
        return ErrorResponse(f"Validation error: {e}", 400)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return ErrorResponse("Internal server error", 500)
```

### Component-Specific Middleware

```python
# Apply middleware to specific components
@mcp.tool(middleware=[rate_limit_middleware, cache_middleware])
async def expensive_operation(data: str) -> dict:
    """Operation with specific middleware."""
    return await process_expensive(data)

# Middleware for all tools
@mcp.tool_middleware()
async def tool_validation_middleware(tool_name, args, next):
    """Validate tool arguments."""
    # Custom validation logic
    if not validate_args(tool_name, args):
        raise ValueError(f"Invalid arguments for {tool_name}")
    
    return await next(args)

# Middleware for resources
@mcp.resource_middleware()
async def resource_cache_middleware(uri, next):
    """Cache resource responses."""
    cached = cache.get(uri)
    if cached:
        return cached
    
    result = await next()
    cache.set(uri, result, ttl=300)
    return result
```

## Advanced Context Management

Beyond basic context variables, FastMCP supports advanced context patterns for complex state management.

### Hierarchical Context

```python
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class HierarchicalContext:
    """Multi-level context with inheritance."""
    global_context: Dict[str, Any] = field(default_factory=dict)
    user_context: Dict[str, Any] = field(default_factory=dict)
    request_context: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, level: str = None) -> Any:
        """Get value from specific level or search hierarchy."""
        if level:
            return getattr(self, f"{level}_context").get(key)
        
        # Search from most specific to least specific
        for ctx_level in ["request", "user", "global"]:
            ctx = getattr(self, f"{ctx_level}_context")
            if key in ctx:
                return ctx[key]
        return None
    
    def set(self, key: str, value: Any, level: str = "request"):
        """Set value at specific level."""
        getattr(self, f"{level}_context")[key] = value

# Context variable
ctx: ContextVar[HierarchicalContext] = ContextVar("hierarchical_context")

@mcp.middleware()
async def setup_context(request, next):
    """Initialize hierarchical context."""
    context = HierarchicalContext()
    
    # Global context (shared across all requests)
    context.global_context = {
        "app_version": "1.0.0",
        "environment": "production"
    }
    
    # User context (per user session)
    user = await get_user_from_request(request)
    if user:
        context.user_context = {
            "user_id": user.id,
            "permissions": user.permissions,
            "preferences": user.preferences
        }
    
    # Request context (per request)
    context.request_context = {
        "request_id": request.id,
        "timestamp": time.time(),
        "ip_address": request.client.host
    }
    
    ctx.set(context)
    return await next(request)
```

### Context Propagation

```python
import asyncio
from contextvars import copy_context

@mcp.tool()
async def parallel_processing(items: list) -> list:
    """Process items in parallel with context propagation."""
    context = copy_context()
    
    async def process_with_context(item):
        # Run in copied context
        return await context.run(process_item, item)
    
    # Process in parallel while maintaining context
    tasks = [process_with_context(item) for item in items]
    results = await asyncio.gather(*tasks)
    
    return results

async def process_item(item):
    """Process item with access to context."""
    ctx_data = ctx.get()
    user_id = ctx_data.get("user_id", "user")
    
    # Context is available in child tasks
    result = await transform_item(item)
    await log_processing(user_id, item, result)
    
    return result
```

## WebSocket Transport

Experimental WebSocket transport for real-time bidirectional communication.

```python
from fastmcp.transports import WebSocketTransport

# Server with WebSocket support
mcp = FastMCP("WebSocket Server")

@mcp.tool()
async def subscribe_updates(topic: str, ctx: Context) -> None:
    """Subscribe to real-time updates."""
    async def send_updates():
        while True:
            update = await get_next_update(topic)
            await ctx.send_message({
                "type": "update",
                "topic": topic,
                "data": update
            })
    
    # Start background task
    asyncio.create_task(send_updates())
    return {"status": "subscribed", "topic": topic}

# Run with WebSocket transport
if __name__ == "__main__":
    transport = WebSocketTransport(host="0.0.0.0", port=8080)
    mcp.run(transport=transport)
```

### WebSocket Client

```python
from fastmcp import Client
from fastmcp.transports import WebSocketTransport

async def handle_messages(message):
    """Handle incoming WebSocket messages."""
    if message["type"] == "update":
        print(f"Update for {message['topic']}: {message['data']}")

# Connect with WebSocket
transport = WebSocketTransport(url="ws://localhost:8080")
client = Client(transport, message_handler=handle_messages)

async with client:
    # Subscribe to updates
    await client.call_tool("subscribe_updates", {"topic": "news"})
    
    # Keep connection alive
    await asyncio.sleep(3600)
```

## Streaming Responses

Support for streaming large responses or real-time data.

```python
from fastmcp import FastMCP, StreamingResponse
import asyncio

mcp = FastMCP("Streaming Server")

@mcp.tool()
async def stream_logs(service: str, lines: int = 100) -> StreamingResponse:
    """Stream logs in real-time."""
    
    async def log_generator():
        # Stream historical logs
        async for line in read_log_lines(service, lines):
            yield f"data: {line}\n\n"
        
        # Stream new logs as they arrive
        async for line in tail_logs(service):
            yield f"data: {line}\n\n"
    
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream"
    )

@mcp.tool()
async def generate_report(params: dict) -> StreamingResponse:
    """Generate large report with progress updates."""
    
    async def report_generator():
        yield '{"status": "starting"}\n'
        
        # Process in chunks
        total = await count_items(params)
        processed = 0
        
        async for chunk in process_chunks(params):
            processed += len(chunk)
            progress = (processed / total) * 100
            
            yield json.dumps({
                "status": "processing",
                "progress": progress,
                "data": chunk
            }) + "\n"
        
        yield '{"status": "complete"}\n'
    
    return StreamingResponse(
        report_generator(),
        media_type="application/x-ndjson"
    )
```

## Plugin System

Experimental plugin architecture for extending FastMCP functionality.

```python
from fastmcp.plugins import Plugin, PluginManager

class MetricsPlugin(Plugin):
    """Plugin for collecting metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    async def on_tool_call(self, tool_name: str, args: dict, result: Any):
        """Hook called after tool execution."""
        if tool_name not in self.metrics:
            self.metrics[tool_name] = {"calls": 0, "errors": 0}
        
        self.metrics[tool_name]["calls"] += 1
        if isinstance(result, Exception):
            self.metrics[tool_name]["errors"] += 1
    
    async def on_server_start(self, server: FastMCP):
        """Hook called when server starts."""
        # Register metrics endpoint
        @server.resource("metrics://stats")
        def get_metrics():
            return self.metrics

class CachingPlugin(Plugin):
    """Plugin for automatic caching."""
    
    def __init__(self, ttl=300):
        self.cache = {}
        self.ttl = ttl
    
    async def before_tool_call(self, tool_name: str, args: dict) -> Any:
        """Hook called before tool execution."""
        cache_key = f"{tool_name}:{hash(str(args))}"
        
        if cache_key in self.cache:
            age = time.time() - self.cache[cache_key]["time"]
            if age < self.ttl:
                return self.cache[cache_key]["result"]
        
        return None  # Continue with normal execution
    
    async def after_tool_call(self, tool_name: str, args: dict, result: Any):
        """Cache successful results."""
        if not isinstance(result, Exception):
            cache_key = f"{tool_name}:{hash(str(args))}"
            self.cache[cache_key] = {
                "result": result,
                "time": time.time()
            }

# Use plugins
mcp = FastMCP("Server with Plugins")
plugin_manager = PluginManager()

plugin_manager.register(MetricsPlugin())
plugin_manager.register(CachingPlugin(ttl=600))

mcp.use_plugin_manager(plugin_manager)
```

## Distributed Tracing

Integration with OpenTelemetry for distributed tracing across MCP services.

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure exporter
otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Tracing middleware
@mcp.middleware()
async def tracing_middleware(request, next):
    """Add distributed tracing."""
    with tracer.start_as_current_span(
        f"{request.method} {request.path}",
        attributes={
            "mcp.request.id": request.id,
            "mcp.request.method": request.method,
            "mcp.request.path": request.path
        }
    ) as span:
        try:
            response = await next(request)
            span.set_attribute("mcp.response.status", response.status)
            return response
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR))
            raise

# Trace tool executions
@mcp.tool()
async def traced_operation(data: str) -> dict:
    """Operation with tracing."""
    with tracer.start_as_current_span("process_data") as span:
        span.set_attribute("data.length", len(data))
        
        # Trace sub-operations
        with tracer.start_as_current_span("validate"):
            validated = await validate_data(data)
        
        with tracer.start_as_current_span("transform"):
            transformed = await transform_data(validated)
        
        with tracer.start_as_current_span("save"):
            result = await save_data(transformed)
        
        span.set_attribute("result.id", result["id"])
        return result
```

## GraphQL Integration

Experimental GraphQL integration for MCP servers.

```python
from fastmcp.integrations import GraphQLIntegration
import strawberry

# Define GraphQL schema
@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: str) -> dict:
        # Call MCP tool
        return await mcp.call_tool("get_user", {"id": id})
    
    @strawberry.field
    async def products(self, category: str = None) -> list:
        return await mcp.call_tool("list_products", {"category": category})

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_user(self, name: str, email: str) -> dict:
        return await mcp.call_tool("create_user", {
            "name": name,
            "email": email
        })

# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)

# Integrate with MCP
mcp = FastMCP("GraphQL Server")
graphql = GraphQLIntegration(schema)
mcp.use_integration(graphql)

# MCP tools are now accessible via GraphQL
@mcp.tool()
async def get_user(id: str) -> dict:
    return {"id": id, "name": "User Name"}

@mcp.tool()
async def list_products(category: str = None) -> list:
    products = [{"id": 1, "name": "Product 1"}]
    if category:
        products = [p for p in products if p.get("category") == category]
    return products
```

## Feature Flags

Experimental feature flag system for gradual rollouts.

```python
from fastmcp.experimental import FeatureFlags

flags = FeatureFlags({
    "new_algorithm": {
        "enabled": True,
        "rollout_percentage": 10,
        "allowlist": ["user123", "user456"]
    },
    "premium_features": {
        "enabled": True,
        "requires_permission": "premium"
    }
})

@mcp.tool()
async def process_data(data: str, ctx: Context) -> dict:
    """Process with feature flags."""
    user_id = ctx.get("user_id")
    
    if flags.is_enabled("new_algorithm", user_id=user_id):
        # Use new algorithm
        result = await new_processing_algorithm(data)
    else:
        # Use old algorithm
        result = await legacy_processing(data)
    
    # Check premium features
    if flags.is_enabled("premium_features", permissions=ctx.get("permissions")):
        result["premium_analysis"] = await premium_analysis(data)
    
    return result

# Dynamic feature flag updates
@mcp.resource("config://feature_flags")
async def get_feature_flags() -> dict:
    """Get current feature flag configuration."""
    return flags.get_all()

@mcp.tool()
async def update_feature_flag(name: str, config: dict) -> dict:
    """Update feature flag configuration."""
    flags.update(name, config)
    return {"status": "updated", "flag": name}
```

## Summary

These experimental features showcase the extensibility and flexibility of FastMCP:

- **Server Composition** - Build modular, reusable MCP services
- **Middleware System** - Intercept and modify request/response flow
- **Advanced Context** - Sophisticated state management patterns
- **WebSocket Transport** - Real-time bidirectional communication
- **Streaming Responses** - Handle large data efficiently
- **Plugin System** - Extend FastMCP with custom functionality
- **Distributed Tracing** - Monitor and debug distributed systems
- **GraphQL Integration** - Alternative API paradigm support

Remember that these features are experimental and may change. Always test thoroughly before using in production environments.