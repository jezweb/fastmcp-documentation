# FastMCP Advanced Features Guide

This document covers advanced FastMCP v2 features that enhance server capabilities beyond basic tools and resources.

## Table of Contents
1. [Resource Templates](#resource-templates)
2. [Elicitation (Interactive Input)](#elicitation-interactive-input)
3. [Progress Tracking](#progress-tracking)
4. [Sampling (LLM Integration)](#sampling-llm-integration)
5. [OpenAPI Integration](#openapi-integration)
6. [Tool Transformation](#tool-transformation)
7. [Authentication Patterns](#authentication-patterns)
8. [Client Configuration](#client-configuration)
9. [Context Management](#context-management)
10. [Server-Side Logging](#server-side-logging)
11. [Proxy Server](#proxy-server)

## Resource Templates

Resource templates are dynamic resources that generate content based on parameters. They're perfect for creating reusable data patterns with variable inputs.

### Basic Resource Template
```python
from fastmcp import FastMCP

mcp = FastMCP("template-server")

# Template with single parameter
@mcp.resource("user://{user_id}/profile")
async def get_user_profile(user_id: str) -> dict:
    """Dynamic user profile resource."""
    user = await fetch_user(user_id)
    return {
        "id": user_id,
        "name": user.name,
        "email": user.email
    }

# Template with multiple parameters
@mcp.resource("report://{type}/{date}/summary")
async def get_report(type: str, date: str) -> dict:
    """Generate report for specific type and date."""
    data = await fetch_report_data(type, date)
    return {
        "type": type,
        "date": date,
        "summary": data
    }
```

### Client Usage
```python
from fastmcp import Client

async with Client("server.py") as client:
    # List available templates
    templates = await client.list_resource_templates()
    
    # Read from template with parameters
    user_profile = await client.read_resource("user://123/profile")
    report = await client.read_resource("report://sales/2025-01-20/summary")
```

### Use Cases
- User-specific data retrieval
- Date-based reports
- Dynamic configuration by environment
- Parameterized data queries

## Elicitation (Interactive Input)

**Requirements:** MCP spec version 2.10.0+

Elicitation allows servers to request structured input from users during tool execution. This enables interactive workflows and dynamic parameter gathering.

### Server Implementation
```python
from fastmcp import FastMCP, Context
from dataclasses import dataclass
from typing import Literal

mcp = FastMCP("interactive-server")

@dataclass
class ProjectConfig:
    name: str
    type: Literal["web", "api", "cli"]
    description: str

@mcp.tool()
async def create_project(ctx: Context) -> str:
    """Create project with interactive parameter gathering."""
    
    # Request structured input from the client
    result = await ctx.elicit(
        "Please provide project configuration:",
        response_type=ProjectConfig
    )
    
    if result.action == "accept":
        config = result.data
        # Create project with provided configuration
        return f"Created {config.type} project: {config.name}"
    elif result.action == "decline":
        return "Project creation declined by user"
    else:
        return "Project creation cancelled"

@mcp.tool()
async def confirm_deletion(file_path: str, ctx: Context) -> str:
    """Delete file with user confirmation."""
    
    # Simple confirmation (expects empty response)
    result = await ctx.elicit(
        f"Are you sure you want to delete {file_path}?",
        response_type=None  # Just confirmation
    )
    
    if result.action == "accept":
        # Perform deletion
        return f"Deleted {file_path}"
    else:
        return "Deletion cancelled"
```

### Client Handler Implementation
```python
from fastmcp import Client

async def elicitation_handler(
    message: str, 
    response_type: type, 
    params, 
    context
):
    """Handle elicitation requests from server."""
    print(f"Server requests: {message}")
    
    # response_type is a dataclass created from the server's schema
    if response_type:
        # Collect user input
        if hasattr(response_type, '__annotations__'):
            # Build response based on dataclass fields
            kwargs = {}
            for field, field_type in response_type.__annotations__.items():
                value = input(f"{field}: ")
                kwargs[field] = value
            return response_type(**kwargs)
    else:
        # Simple confirmation
        confirm = input("Confirm? (y/n): ")
        if confirm.lower() == 'y':
            return {}  # Empty object for confirmation
        else:
            return None  # Indicates decline

# Create client with handler
client = Client("server.py", elicitation_handler=elicitation_handler)

async with client:
    result = await client.call_tool("create_project")
```

### Use Cases
- Missing parameter collection
- User confirmation for sensitive operations
- Dynamic workflow decisions
- Multi-step wizards

## Progress Tracking

**Requirements:** Client must provide `progressToken` in request

Progress tracking enables monitoring of long-running operations, essential for user feedback during batch processing or slow operations. Note: If the client doesn't provide a `progressToken`, progress reporting calls will be silently ignored.

### Server Implementation
```python
from fastmcp import FastMCP, Context
import asyncio

mcp = FastMCP("progress-server")

@mcp.tool()
async def batch_process(items: list[str], ctx: Context) -> dict:
    """Process items with progress reporting."""
    results = []
    total = len(items)
    
    for i, item in enumerate(items):
        # Report progress to client
        await ctx.report_progress(
            progress=i,
            total=total
        )
        
        # Simulate processing
        await asyncio.sleep(0.1)
        result = item.upper()
        results.append(result)
    
    # Report completion
    await ctx.report_progress(progress=total, total=total)
    
    return {"processed": total, "results": results}

@mcp.tool()
async def multi_stage_operation(ctx: Context) -> str:
    """Operation with percentage-based progress."""
    stages = [
        ("Initializing", 25),
        ("Processing data", 50),
        ("Analyzing results", 75),
        ("Generating report", 100)
    ]
    
    for stage_name, percentage in stages:
        await ctx.info(f"Stage: {stage_name}")
        await ctx.report_progress(
            progress=percentage,
            total=100
        )
        await asyncio.sleep(0.5)  # Simulate work
    
    return "Operation completed successfully"

@mcp.tool()
async def indeterminate_scan(directory: str, ctx: Context) -> dict:
    """Scan with unknown total items."""
    files_found = 0
    
    # Simulate scanning unknown number of files
    for _ in range(10):  # Unknown at start
        files_found += 1
        
        # Progress without total for indeterminate operations
        await ctx.report_progress(progress=files_found)
        
        await asyncio.sleep(0.2)
    
    return {"files_found": files_found, "directory": directory}
```

### Client Handler
```python
async def progress_handler(progress: float, total: float | None, message: str | None):
    """Handle progress updates from server."""
    if total:
        percentage = (progress / total) * 100
        print(f"[{percentage:.1f}%] {message or 'Processing...'}")
    else:
        print(f"Progress: {progress} - {message or ''}")

client = Client("server.py", progress_handler=progress_handler)

async with client:
    # Progress updates will be displayed during execution
    result = await client.call_tool("batch_process", {"items": large_list})
```

### Use Cases
- Batch operations
- File uploads/downloads
- Data migrations
- Complex calculations
- API bulk operations

## Sampling (LLM Integration)

**Requirements:** MCP spec version 2.0.0+

Sampling allows MCP servers to request LLM completions from clients, enabling AI-powered validation, content generation, and decision-making within tools.

### Server Implementation
```python
from fastmcp import FastMCP, Context

mcp = FastMCP("ai-powered-server")

@mcp.tool()
async def analyze_sentiment(text: str, ctx: Context) -> dict:
    """Analyze sentiment using client's LLM."""
    
    # Simple text prompt
    response = await ctx.sample(
        f"Analyze the sentiment of this text as positive, negative, or neutral. "
        f"Reply with just one word.\n\nText: {text}",
        temperature=0.3
    )
    
    sentiment = response.text.strip().lower()
    return {"text": text, "sentiment": sentiment}

@mcp.tool()
async def generate_description(product: dict, ctx: Context) -> str:
    """Generate product description using LLM with parameters."""
    
    # Create the prompt
    prompt = f"""Generate a compelling product description for:
    Name: {product['name']}
    Category: {product['category']}
    Features: {', '.join(product['features'])}
    """
    
    # Request with system prompt and parameters
    response = await ctx.sample(
        messages=[{"role": "user", "content": prompt}],
        modelPreferences={
            "temperature": 0.7,
            "maxTokens": 200,
            "systemPrompt": "You are a creative product description writer."
        }
    )
    
    return response.text

@mcp.tool()
async def validate_content(text: str, ctx: Context) -> dict:
    """Validate content using AI."""
    
    validation_prompt = f"Analyze this text for issues: {text}"
    
    response = await ctx.sample(
        messages=[{"role": "user", "content": validation_prompt}],
        modelPreferences={
            "systemPrompt": "You are a content validator. Check for errors, bias, and clarity.",
            "temperature": 0.3
        }
    )
    
    return {"original": text, "analysis": response.text}
```

### Client Handler with Gemini
```python
from fastmcp import Client
import google.generativeai as genai

async def sampling_handler(messages, model_preferences):
    """Handle sampling requests using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash-001')
    
    # Convert messages to Gemini format
    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
    
    # Add system prompt if provided
    if model_preferences and model_preferences.get('systemPrompt'):
        prompt = f"{model_preferences['systemPrompt']}\n\n{prompt}"
    
    # Extract generation config
    generation_config = {}
    if model_preferences:
        if 'temperature' in model_preferences:
            generation_config['temperature'] = model_preferences['temperature']
        if 'maxTokens' in model_preferences:
            generation_config['max_output_tokens'] = model_preferences['maxTokens']
    
    # Generate response
    response = model.generate_content(
        prompt,
        generation_config=generation_config or None
    )
    
    return {"text": response.text}

client = Client("server.py", sampling_handler=sampling_handler)
```

### Use Cases
- Content generation
- Data validation
- Intelligent decision-making
- Natural language processing
- Code generation within tools

## OpenAPI Integration

FastMCP can automatically generate MCP servers from OpenAPI/Swagger specifications, instantly making any REST API accessible to LLMs.

### Basic OpenAPI Integration
```python
import httpx
from fastmcp import FastMCP

# Load OpenAPI specification
spec_url = "https://api.example.com/openapi.json"
spec = httpx.get(spec_url).json()

# Create authenticated client
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {API_KEY}"}
)

# Generate MCP server from OpenAPI
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    name="API Server"
)

# All endpoints are now MCP tools/resources!
```

### Advanced Route Mapping
```python
from fastmcp.server.openapi import RouteMap, MCPType

# Customize how endpoints map to MCP components
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
    
    # Mutations → Tools
    RouteMap(
        methods=["POST", "PUT", "PATCH", "DELETE"],
        mcp_type=MCPType.TOOL
    ),
    
    # Exclude internal endpoints
    RouteMap(
        pattern=r"/internal/.*",
        mcp_type=MCPType.EXCLUDE
    )
]

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    route_maps=route_maps,
    name="Custom API Server"
)
```

### Component Customization
```python
def customize_component(component, route_info):
    """Customize generated components."""
    # Improve names
    if component.name.startswith("get_"):
        component.name = component.name[4:]
    
    # Add tags
    if "/admin/" in route_info.path:
        component.tags = ["admin", "restricted"]
    
    # Enhance descriptions
    if not component.description:
        component.description = f"Access {route_info.path}"
    
    return component

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    mcp_component_fn=customize_component
)
```

### Use Cases
- Instant API wrapper creation
- Legacy system integration
- Third-party service connection
- Rapid prototyping

## Tool Transformation

Tool transformation allows creating enhanced variants of existing tools with modified behavior, validation, or parameters.

### Basic Transformation
```python
from fastmcp import Tool, ArgTransform

# Original tool
@mcp.tool()
def original_search(query: str, limit: int = 10) -> dict:
    """Basic search tool."""
    return perform_search(query, limit)

# Create enhanced version
enhanced_search = Tool.from_tool(
    original_search,
    name="smart_search",
    description="Enhanced search with defaults and validation",
    transform_args={
        "query": ArgTransform(
            description="Search query (automatically cleaned)",
            hide=False
        ),
        "limit": ArgTransform(
            default=20,  # Change default
            description="Number of results (max 100)"
        )
    }
)

mcp.add_tool(enhanced_search)
```

### Advanced Transformation with Validation
```python
from fastmcp import forward

async def validate_and_transform(query: str, limit: int) -> dict:
    """Add validation and preprocessing."""
    # Clean query
    query = query.strip().lower()
    
    # Validate
    if len(query) < 3:
        return {"error": "Query too short (min 3 characters)"}
    
    if limit > 100:
        limit = 100  # Cap at maximum
    
    # Call original with transformed params
    return await forward(query=query, limit=limit)

validated_search = Tool.from_tool(
    original_search,
    name="validated_search",
    transform_fn=validate_and_transform
)
```

### Context-Aware Tools
```python
def create_user_tool(user_context):
    """Factory for user-specific tools."""
    
    async def user_aware_operation(action: str) -> dict:
        # Automatically include user context
        return await perform_action(
            action=action,
            user_id=user_context.id,
            permissions=user_context.permissions
        )
    
    return Tool.from_tool(
        generic_operation,
        name=f"user_{user_context.id}_operation",
        transform_fn=user_aware_operation,
        transform_args={
            "user_id": ArgTransform(hide=True, default=user_context.id)
        }
    )

# Create user-specific tool
user_tool = create_user_tool(current_user)
mcp.add_tool(user_tool)
```

### Use Cases
- Simplifying complex interfaces
- Adding validation layers
- Creating user-specific tools
- Hiding sensitive parameters
- Setting environment-specific defaults

## Authentication Patterns

### OAuth2 Authentication
```python
from fastmcp.server.auth import OAuth2Provider

oauth = OAuth2Provider(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    authorize_url="https://auth.example.com/authorize",
    token_url="https://auth.example.com/token",
    redirect_uri="http://localhost:8000/callback"
)

mcp = FastMCP("oauth-server", auth=oauth)
```

### Bearer Token Authentication
```python
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    issuer="https://auth.example.com",
    audience="my-server",
    algorithms=["RS256"]
)

mcp = FastMCP("secure-server", auth=auth)
```

### Custom Authentication
```python
class APIKeyAuth:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def authenticate(self, request):
        """Validate API key from request."""
        provided_key = request.headers.get("X-API-Key")
        if provided_key != self.api_key:
            raise AuthenticationError("Invalid API key")
        return True

auth = APIKeyAuth(os.getenv("API_KEY"))
mcp = FastMCP("api-key-server", auth=auth)
```

## Client Configuration

### Advanced Client Setup
```python
from fastmcp import Client

# Full configuration
client = Client(
    "server.py",
    
    # Handlers
    elicitation_handler=handle_elicitation,
    progress_handler=handle_progress,
    sampling_handler=handle_sampling,
    
    # Logging
    log_level="DEBUG",
    log_file="mcp_client.log",
    
    # Connection
    timeout=60,
    max_retries=3,
    
    # Messages
    message_handler=handle_messages,
    
    # Roots (working directories)
    roots=["/home/user/projects", "/tmp/mcp"]
)
```

### Message Handling
```python
async def message_handler(message_type: str, content: dict):
    """Handle server messages."""
    if message_type == "error":
        logger.error(f"Server error: {content}")
    elif message_type == "warning":
        logger.warning(f"Server warning: {content}")
    elif message_type == "info":
        logger.info(f"Server info: {content}")
    
    # Can also display to user
    if content.get("user_visible"):
        print(f"Server: {content['message']}")
```

### Roots Configuration
```python
# Roots define accessible directories for file operations
client = Client(
    "server.py",
    roots=[
        {"path": "/home/user/data", "name": "data"},
        {"path": "/tmp/processing", "name": "temp"}
    ]
)

# Server can then access files within these roots
@mcp.tool()
async def process_file(root: str, filename: str):
    """Process file from allowed roots."""
    # Client ensures file access is within defined roots
    file_path = get_root_path(root) / filename
    return process(file_path)
```

## Integration Priority Guide

Based on typical use cases, here's the recommended adoption order:

### Phase 1: Immediate Value (Start Here)
1. **OpenAPI Integration** - Instant API access
2. **Tool Transformation** - Simplify existing tools
3. **Progress Tracking** - Better UX for long operations

### Phase 2: Enhanced Functionality
4. **Resource Templates** - Dynamic data access
5. **Authentication** - Secure API access
6. **Client Configuration** - Advanced logging/debugging

### Phase 3: Advanced Features
7. **Elicitation** - Interactive workflows
8. **Sampling** - AI-powered operations

## Example: Complete Advanced Server

```python
from fastmcp import FastMCP, Tool, forward
import httpx
import os

# Create server with auth
mcp = FastMCP("advanced-server")

# Load and integrate OpenAPI
spec = httpx.get("https://api.example.com/openapi.json").json()
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {os.getenv('API_KEY')}"}
)

# Generate tools from OpenAPI
api_tools = FastMCP.from_openapi(spec, client)
for tool in api_tools.tools:
    mcp.add_tool(tool)

# Add resource template
@mcp.resource("data://{category}/{id}")
async def get_data(category: str, id: str) -> dict:
    """Dynamic data retrieval."""
    return await fetch_data(category, id)

# Transform a complex tool
simple_tool = Tool.from_tool(
    api_tools.get_tool("complex_operation"),
    name="simple_op",
    transform_args={
        "auth_token": {"hide": True, "default": os.getenv("TOKEN")},
        "verbose": {"hide": True, "default": False}
    }
)
mcp.add_tool(simple_tool)

# Add progress tracking
@mcp.tool()
async def batch_operation(items: list, ctx: Context) -> dict:
    """Process with progress."""
    for i, item in enumerate(items):
        await ctx.report_progress(
            progress=i+1,
            total=len(items),
            message=f"Processing {item}"
        )
        await process(item)
    return {"status": "complete"}

if __name__ == "__main__":
    mcp.run()
```

## Context Management

Context management in FastMCP provides request-scoped state sharing using Python's `contextvars` module. This enables clean data flow between middleware, tools, and resources without parameter passing.

### Basic Context Variables
```python
from fastmcp import FastMCP
from contextvars import ContextVar
import asyncio

# Define context variables
current_user: ContextVar[dict] = ContextVar('current_user')
request_id: ContextVar[str] = ContextVar('request_id')
session_data: ContextVar[dict] = ContextVar('session_data')

mcp = FastMCP("context-server")

@mcp.middleware()
async def context_middleware(request, next):
    """Initialize request context."""
    # Set request-scoped variables
    request_id.set(f"req_{asyncio.current_task().get_name()}")
    current_user.set({"id": "user123", "role": "admin"})
    session_data.set({"last_action": "login", "preferences": {}})
    
    return await next(request)

@mcp.tool()
async def get_user_data() -> dict:
    """Access context without parameters."""
    user = current_user.get()
    req_id = request_id.get()
    
    return {
        "user_id": user["id"],
        "request_id": req_id,
        "authorized": user["role"] == "admin"
    }

@mcp.resource("user://profile")
async def user_profile() -> dict:
    """Resource using context."""
    user = current_user.get()
    session = session_data.get()
    
    return {
        "user": user,
        "last_action": session["last_action"]
    }
```

### Advanced Context Patterns
```python
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from typing import Optional
import structlog

# Structured context data
@dataclass
class RequestContext:
    request_id: str
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    feature_flags: dict = None
    start_time: float = None

# Context variable
ctx: ContextVar[RequestContext] = ContextVar('request_context')

# Context manager for easy access
class Context:
    @staticmethod
    def get() -> RequestContext:
        return ctx.get()
    
    @staticmethod
    def set(context: RequestContext):
        ctx.set(context)
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        return ctx.get().user_id
    
    @staticmethod
    def get_request_id() -> str:
        return ctx.get().request_id

@mcp.middleware()
async def enhanced_context_middleware(request, next):
    """Enhanced context initialization."""
    import time
    import uuid
    
    # Create comprehensive context
    context = RequestContext(
        request_id=str(uuid.uuid4()),
        user_id=request.get("user_id"),
        organization_id=request.get("org_id"),
        feature_flags=await load_feature_flags(request.get("user_id")),
        start_time=time.time()
    )
    
    Context.set(context)
    
    # Add to structured logging
    structlog.contextvars.bind_contextvars(
        request_id=context.request_id,
        user_id=context.user_id
    )
    
    try:
        return await next(request)
    finally:
        # Log request completion
        elapsed = time.time() - context.start_time
        logger = structlog.get_logger()
        logger.info("Request completed", elapsed_seconds=elapsed)
```

### Database Session Context
```python
from sqlalchemy.ext.asyncio import AsyncSession
from contextvars import ContextVar

db_session: ContextVar[AsyncSession] = ContextVar('db_session')

@mcp.middleware()
async def database_context_middleware(request, next):
    """Provide database session in context."""
    async with AsyncSessionLocal() as session:
        db_session.set(session)
        try:
            response = await next(request)
            await session.commit()
            return response
        except Exception:
            await session.rollback()
            raise

@mcp.tool()
async def create_user(name: str, email: str) -> dict:
    """Create user using context session."""
    session = db_session.get()
    
    user = User(name=name, email=email)
    session.add(user)
    await session.flush()  # Get ID without committing
    
    return {"id": user.id, "name": user.name}
```

### Propagating Context to Background Tasks
```python
import asyncio
from contextvars import copy_context

@mcp.tool()
async def schedule_background_work(task_data: dict) -> dict:
    """Schedule work that preserves context."""
    # Copy current context
    current_context = copy_context()
    
    async def background_task():
        """Background task with preserved context."""
        # Context variables are available here
        user = current_user.get()
        req_id = request_id.get()
        
        await perform_background_work(task_data, user, req_id)
    
    # Run in copied context
    asyncio.create_task(background_task(), context=current_context)
    
    return {"status": "scheduled", "request_id": request_id.get()}
```

### Use Cases
- User session management
- Request tracing and correlation IDs
- Database session sharing
- Feature flag access
- Audit logging context

## Server-Side Logging

FastMCP provides comprehensive logging capabilities that can stream log messages directly to clients, enabling real-time monitoring and debugging.

### Basic Client Logging
```python
from fastmcp import FastMCP
from fastmcp.logging import ClientLogger
import structlog

mcp = FastMCP("logging-server")

# Enable client logging
client_logger = ClientLogger(mcp)
logger = structlog.get_logger()

@mcp.tool()
async def process_data(data: dict) -> dict:
    """Tool with client-visible logging."""
    
    # These logs go to the client
    await client_logger.info("Starting data processing", data_size=len(data))
    
    try:
        # Process data with progress logging
        for i, item in enumerate(data.get("items", [])):
            await client_logger.debug(f"Processing item {i+1}", item_id=item.get("id"))
            result = await process_item(item)
            
        await client_logger.info("Data processing completed successfully")
        return {"status": "success", "processed": len(data.get("items", []))}
        
    except Exception as e:
        await client_logger.error("Processing failed", error=str(e), exc_info=True)
        raise
```

### Structured Logging Configuration
```python
import structlog
from fastmcp.logging import setup_client_logging

# Configure structured logging for client streaming
setup_client_logging(
    level="INFO",
    format="json",
    stream_to_client=True,
    buffer_size=100,  # Buffer messages for batching
    flush_interval=1.0,  # Flush every second
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ]
)

mcp = FastMCP("structured-logging-server")

@mcp.tool()
async def analyze_file(file_path: str) -> dict:
    """File analysis with structured logging."""
    logger = structlog.get_logger()
    
    # Structured log with context
    await logger.ainfo(
        "File analysis started",
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        request_id=request_id.get()
    )
    
    results = []
    async for chunk in read_file_chunks(file_path):
        chunk_result = await analyze_chunk(chunk)
        results.append(chunk_result)
        
        # Progress logging
        await logger.adebug(
            "Chunk processed",
            chunk_number=len(results),
            chunk_size=len(chunk),
            patterns_found=len(chunk_result.get("patterns", []))
        )
    
    await logger.ainfo(
        "File analysis completed",
        total_chunks=len(results),
        patterns_found=sum(len(r.get("patterns", [])) for r in results)
    )
    
    return {"results": results}
```

### Log Level Management
```python
from fastmcp.logging import LogLevel, DynamicLogger

# Create logger with dynamic level control
dynamic_logger = DynamicLogger(mcp, default_level=LogLevel.INFO)

@mcp.tool()
async def set_log_level(level: str) -> dict:
    """Change logging level at runtime."""
    log_level = LogLevel.from_string(level.upper())
    dynamic_logger.set_level(log_level)
    
    await dynamic_logger.info(f"Log level changed to {level.upper()}")
    return {"level": level, "status": "updated"}

@mcp.tool()
async def debug_operation(data: dict) -> dict:
    """Operation with conditional debug logging."""
    
    # Always log at info level
    await dynamic_logger.info("Debug operation started")
    
    # Debug logs only appear if level is DEBUG
    await dynamic_logger.debug("Input data received", data=data)
    await dynamic_logger.debug("Validation step", valid=validate_data(data))
    
    result = await process_data(data)
    
    await dynamic_logger.debug("Processing result", result=result)
    await dynamic_logger.info("Debug operation completed")
    
    return result
```

### Categorized Logging
```python
from fastmcp.logging import CategoryLogger

# Create category-specific loggers
auth_logger = CategoryLogger(mcp, "auth")
db_logger = CategoryLogger(mcp, "database")
api_logger = CategoryLogger(mcp, "external_api")

@mcp.tool()
async def secure_operation(user_id: str, action: str) -> dict:
    """Multi-category logging example."""
    
    # Authentication logging
    await auth_logger.info("Authentication check", user_id=user_id)
    user = await authenticate_user(user_id)
    if not user:
        await auth_logger.warning("Authentication failed", user_id=user_id)
        raise AuthenticationError("Invalid user")
    
    # Database logging
    await db_logger.debug("Database query starting", table="users", user_id=user_id)
    user_data = await fetch_user_data(user_id)
    await db_logger.debug("Database query completed", records_returned=len(user_data))
    
    # External API logging
    await api_logger.info("External API call", endpoint="/api/process", action=action)
    try:
        api_result = await call_external_api(action, user_data)
        await api_logger.info("External API success", response_size=len(api_result))
    except Exception as e:
        await api_logger.error("External API failed", error=str(e))
        raise
    
    return {"status": "success", "user": user_data}
```

### Performance Logging
```python
from fastmcp.logging import PerformanceLogger
import time

perf_logger = PerformanceLogger(mcp)

@mcp.tool()
async def performance_sensitive_operation(data: list) -> dict:
    """Operation with performance logging."""
    
    # Start timing
    with perf_logger.timer("total_operation"):
        
        # Individual step timing
        with perf_logger.timer("validation"):
            await validate_input(data)
        
        results = []
        with perf_logger.timer("processing"):
            for item in data:
                with perf_logger.timer("item_processing"):
                    result = await process_item(item)
                    results.append(result)
        
        with perf_logger.timer("finalization"):
            final_result = await finalize_results(results)
    
    # Log performance summary
    await perf_logger.info(
        "Performance summary",
        total_items=len(data),
        processing_rate=len(data) / perf_logger.get_elapsed("processing")
    )
    
    return final_result
```

### Client-Side Log Handling
```python
from fastmcp import Client
import structlog

async def log_handler(log_entry: dict):
    """Handle logs from server."""
    level = log_entry.get("level", "info")
    message = log_entry.get("message", "")
    context = log_entry.get("context", {})
    
    # Use client logger
    client_logger = structlog.get_logger("mcp_server")
    
    if level == "debug":
        client_logger.debug(message, **context)
    elif level == "info":
        client_logger.info(message, **context)
    elif level == "warning":
        client_logger.warning(message, **context)
    elif level == "error":
        client_logger.error(message, **context)
    
    # Could also display in UI
    if log_entry.get("user_visible", False):
        print(f"Server: {message}")

client = Client("server.py", log_handler=log_handler)
```

### Use Cases
- Real-time debugging
- Progress monitoring
- Error investigation
- Performance analysis
- Audit trails

## Proxy Server

FastMCP's proxy server enables advanced routing, load balancing, authentication, and aggregation of multiple MCP servers.

### Basic Proxy Setup
```python
from fastmcp.proxy import MCPProxy
from fastmcp.proxy.config import ProxyConfig, ServerConfig

# Configure upstream servers
config = ProxyConfig([
    ServerConfig(
        name="user-service",
        url="http://localhost:8001",
        prefix="/users",
        weight=1
    ),
    ServerConfig(
        name="data-service", 
        url="http://localhost:8002",
        prefix="/data",
        weight=1
    )
])

# Create and run proxy
proxy = MCPProxy(config)
proxy.run(host="0.0.0.0", port=8000)
```

### Advanced Routing Configuration
```python
from fastmcp.proxy import MCPProxy, Route, Matcher
from fastmcp.proxy.routing import PathMatcher, HeaderMatcher, ToolMatcher

proxy = MCPProxy()

# Route by path patterns
proxy.add_route(Route(
    matcher=PathMatcher(pattern="/api/users/*"),
    target="user-service",
    strip_prefix="/api"
))

# Route by tool names
proxy.add_route(Route(
    matcher=ToolMatcher(patterns=["search_*", "query_*"]),
    target="search-service"
))

# Route by headers
proxy.add_route(Route(
    matcher=HeaderMatcher(header="X-Service", value="analytics"),
    target="analytics-service"
))

# Fallback route
proxy.add_route(Route(
    matcher=Matcher.any(),
    target="default-service"
))
```

### Load Balancing Strategies
```python
from fastmcp.proxy.balancing import RoundRobin, WeightedRandom, LeastConnections

# Round-robin load balancing
proxy = MCPProxy(
    servers=[
        ServerConfig("server1", "http://localhost:8001", weight=1),
        ServerConfig("server2", "http://localhost:8002", weight=1),
        ServerConfig("server3", "http://localhost:8003", weight=1)
    ],
    load_balancer=RoundRobin()
)

# Weighted random (for different server capacities)
proxy = MCPProxy(
    servers=[
        ServerConfig("small-server", "http://localhost:8001", weight=1),
        ServerConfig("large-server", "http://localhost:8002", weight=3)  # 3x traffic
    ],
    load_balancer=WeightedRandom()
)

# Least connections (for long-running operations)
proxy = MCPProxy(
    servers=[
        ServerConfig("server1", "http://localhost:8001"),
        ServerConfig("server2", "http://localhost:8002")
    ],
    load_balancer=LeastConnections()
)
```

### Health Checks and Failover
```python
from fastmcp.proxy.health import HealthCheck, HTTPHealthCheck

# Configure health checks
health_check = HTTPHealthCheck(
    path="/health",
    interval=30,  # Check every 30 seconds
    timeout=5,    # 5 second timeout
    retries=3,    # 3 failed checks before marking unhealthy
    success_codes=[200, 204]
)

proxy = MCPProxy(
    servers=[
        ServerConfig(
            "primary", 
            "http://localhost:8001",
            health_check=health_check
        ),
        ServerConfig(
            "backup", 
            "http://localhost:8002", 
            health_check=health_check
        )
    ],
    failover_enabled=True
)

# Custom health check
class CustomHealthCheck(HealthCheck):
    async def check_health(self, server: ServerConfig) -> bool:
        """Custom health check logic."""
        try:
            # Test actual MCP functionality
            async with MCPClient(server.url) as client:
                tools = await client.list_tools()
                return len(tools) > 0
        except Exception:
            return False

proxy = MCPProxy(
    servers=[
        ServerConfig("server1", "http://localhost:8001", 
                    health_check=CustomHealthCheck())
    ]
)
```

### Authentication and Authorization
```python
from fastmcp.proxy.auth import ProxyAuth, JWTAuth, APIKeyAuth

# JWT authentication
jwt_auth = JWTAuth(
    secret_key="your-secret-key",
    algorithms=["HS256"],
    verify_exp=True
)

# API key authentication
api_auth = APIKeyAuth(
    api_keys={"client1": "key1", "client2": "key2"},
    header_name="X-API-Key"
)

# Configure proxy with authentication
proxy = MCPProxy(
    servers=[
        ServerConfig("secure-service", "http://localhost:8001")
    ],
    auth=jwt_auth
)

# Route-specific authentication
proxy.add_route(Route(
    matcher=PathMatcher("/admin/*"),
    target="admin-service",
    auth=api_auth,  # Override proxy auth for admin routes
    required_roles=["admin"]
))
```

### Request/Response Transformation
```python
from fastmcp.proxy.transform import RequestTransformer, ResponseTransformer

class AuthTransformer(RequestTransformer):
    """Add authentication headers to upstream requests."""
    
    async def transform_request(self, request: dict, server: ServerConfig) -> dict:
        # Add server-specific authentication
        if server.name == "user-service":
            request["headers"]["Authorization"] = f"Bearer {USER_SERVICE_TOKEN}"
        elif server.name == "data-service":
            request["headers"]["X-API-Key"] = DATA_SERVICE_KEY
        
        return request

class ResponseTransformer(ResponseTransformer):
    """Transform responses from upstream servers."""
    
    async def transform_response(self, response: dict, server: ServerConfig) -> dict:
        # Add server identification
        response["_server"] = server.name
        response["_timestamp"] = time.time()
        
        # Filter sensitive data
        if "password" in response:
            response.pop("password")
        
        return response

proxy = MCPProxy(
    servers=[...],
    request_transformer=AuthTransformer(),
    response_transformer=ResponseTransformer()
)
```

### Caching Layer
```python
from fastmcp.proxy.cache import RedisCache, MemoryCache

# Redis-based caching
cache = RedisCache(
    url="redis://localhost:6379",
    default_ttl=300,  # 5 minutes
    key_prefix="mcp_proxy:"
)

# Memory-based caching (for development)
cache = MemoryCache(
    max_size=1000,
    default_ttl=300
)

proxy = MCPProxy(
    servers=[...],
    cache=cache,
    cache_rules=[
        # Cache GET requests to resources
        CacheRule(
            matcher=PathMatcher("/resources/*"),
            methods=["GET"],
            ttl=600  # 10 minutes
        ),
        # Cache tool results based on input
        CacheRule(
            matcher=ToolMatcher(["expensive_calculation"]),
            ttl=3600,  # 1 hour
            vary_on=["args"]  # Cache key includes arguments
        )
    ]
)
```

### Rate Limiting
```python
from fastmcp.proxy.ratelimit import RateLimiter, TokenBucket, SlidingWindow

# Token bucket rate limiting
rate_limiter = TokenBucket(
    capacity=100,      # 100 requests
    refill_rate=10,    # 10 requests per second
    refill_period=1    # 1 second
)

# Sliding window rate limiting
rate_limiter = SlidingWindow(
    max_requests=1000,  # 1000 requests
    window_size=3600    # per hour
)

proxy = MCPProxy(
    servers=[...],
    rate_limiter=rate_limiter,
    rate_limit_rules=[
        # Per-client limits
        RateLimit(
            matcher=HeaderMatcher("X-Client-ID", "heavy_user"),
            limit=TokenBucket(capacity=200, refill_rate=20)
        ),
        # Per-endpoint limits
        RateLimit(
            matcher=PathMatcher("/expensive/*"),
            limit=TokenBucket(capacity=10, refill_rate=1)
        )
    ]
)
```

### Monitoring and Metrics
```python
from fastmcp.proxy.monitoring import ProxyMetrics
from prometheus_client import start_http_server

# Enable Prometheus metrics
metrics = ProxyMetrics()
proxy = MCPProxy(servers=[...], metrics=metrics)

# Start metrics server
start_http_server(9090)

# Custom metrics
@proxy.middleware()
async def custom_metrics_middleware(request, next):
    """Add custom metrics collection."""
    start_time = time.time()
    
    try:
        response = await next(request)
        metrics.request_duration.observe(time.time() - start_time)
        metrics.request_count.labels(status="success").inc()
        return response
    except Exception as e:
        metrics.request_count.labels(status="error").inc()
        metrics.error_count.labels(error_type=type(e).__name__).inc()
        raise
```

### Complete Proxy Configuration
```python
from fastmcp.proxy import MCPProxy
from fastmcp.proxy.config import ProxyConfig

# Comprehensive proxy setup
config = ProxyConfig(
    servers=[
        ServerConfig(
            name="user-service",
            url="http://user-service:8001",
            prefix="/users",
            weight=2,
            health_check=HTTPHealthCheck("/health", interval=30),
            max_connections=50
        ),
        ServerConfig(
            name="data-service",
            url="http://data-service:8002", 
            prefix="/data",
            weight=3,
            health_check=HTTPHealthCheck("/health", interval=30),
            max_connections=100
        )
    ],
    
    # Load balancing
    load_balancer=WeightedRandom(),
    
    # Authentication
    auth=JWTAuth(secret_key=os.getenv("JWT_SECRET")),
    
    # Caching
    cache=RedisCache(url=os.getenv("REDIS_URL")),
    
    # Rate limiting
    rate_limiter=TokenBucket(capacity=1000, refill_rate=100),
    
    # Monitoring
    metrics=ProxyMetrics(),
    
    # Timeouts
    upstream_timeout=30,
    connect_timeout=5,
    
    # Retry policy
    retry_attempts=3,
    retry_backoff=1.0,
    
    # Security
    allowed_hosts=["api.example.com"],
    cors_enabled=True,
    cors_origins=["https://app.example.com"]
)

proxy = MCPProxy(config)

if __name__ == "__main__":
    proxy.run(host="0.0.0.0", port=8000)
```

### Use Cases
- Microservices aggregation
- Load balancing and high availability
- API gateway functionality
- Cross-cutting concerns (auth, logging, monitoring)
- Legacy system integration
- Development/staging environment routing

## Resources

- [FastMCP Client Docs](https://gofastmcp.com/clients)
- [OpenAPI Integration](https://gofastmcp.com/integrations/openapi)
- [Gemini SDK Integration](https://gofastmcp.com/integrations/gemini)
- [Tool Transformation](https://gofastmcp.com/patterns/tool-transformation)
- [MCP Protocol Spec](https://modelcontextprotocol.io)