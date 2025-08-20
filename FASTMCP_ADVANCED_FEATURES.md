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

Elicitation allows servers to request structured input from users during tool execution. This enables interactive workflows and dynamic parameter gathering.

### Server Implementation
```python
from fastmcp import FastMCP

mcp = FastMCP("interactive-server")

@mcp.tool()
async def create_project(name: str = None, type: str = None):
    """Create project with interactive parameter gathering."""
    
    # Server can request missing parameters via elicitation
    if not name:
        # This would trigger elicitation on the client
        name = await request_input("project_name", "Enter project name:")
    
    if not type:
        type = await request_input("project_type", 
                                 "Select project type:",
                                 choices=["web", "api", "cli"])
    
    return create_project_structure(name, type)
```

### Client Handler Implementation
```python
from fastmcp import Client
from dataclasses import dataclass

@dataclass
class ProjectInput:
    project_name: str
    project_type: str

async def elicitation_handler(message: str, response_type: type, context: dict):
    """Handle elicitation requests from server."""
    print(f"Server requests: {message}")
    
    # Collect user input based on response_type
    if response_type == ProjectInput:
        name = input("Project name: ")
        type = input("Project type (web/api/cli): ")
        return ProjectInput(project_name=name, project_type=type)
    
    # Can also decline or cancel
    # return ElicitationResponse.decline("User cancelled")

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

Progress tracking enables monitoring of long-running operations, essential for user feedback during batch processing or slow operations.

### Server Implementation
```python
from fastmcp import FastMCP

mcp = FastMCP("progress-server")

@mcp.tool()
async def batch_process(items: list) -> dict:
    """Process items with progress reporting."""
    results = []
    total = len(items)
    
    for i, item in enumerate(items):
        # Report progress to client
        await report_progress(
            progress=i + 1,
            total=total,
            message=f"Processing {item}..."
        )
        
        result = await process_item(item)
        results.append(result)
    
    return {"processed": total, "results": results}

@mcp.tool()
async def long_analysis(data: dict) -> dict:
    """Long-running analysis with stage reporting."""
    stages = ["Validating", "Analyzing", "Generating Report"]
    
    for i, stage in enumerate(stages):
        await report_progress(
            progress=i + 1,
            total=len(stages),
            message=stage
        )
        await perform_stage(stage, data)
    
    return {"status": "complete"}
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

Sampling allows MCP servers to request LLM completions from clients, enabling AI-powered validation, content generation, and decision-making within tools.

### Server Implementation
```python
from fastmcp import FastMCP

mcp = FastMCP("ai-powered-server")

@mcp.tool()
async def generate_description(product: dict) -> str:
    """Generate product description using LLM."""
    
    # Request LLM completion from client
    prompt = f"""Generate a compelling product description for:
    Name: {product['name']}
    Category: {product['category']}
    Features: {', '.join(product['features'])}
    """
    
    description = await request_sampling(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    
    return description

@mcp.tool()
async def validate_content(text: str) -> dict:
    """Validate content using AI."""
    
    validation_prompt = f"Analyze this text for issues: {text}"
    
    analysis = await request_sampling(
        messages=[{"role": "user", "content": validation_prompt}],
        system_prompt="You are a content validator. Check for errors, bias, and clarity."
    )
    
    return {"original": text, "analysis": analysis}
```

### Client Handler with Gemini
```python
from fastmcp import Client
import google.generativeai as genai

async def sampling_handler(messages, params, context):
    """Handle sampling requests using Gemini."""
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Convert messages to Gemini format
    prompt = "\n".join([f"{m.role}: {m.content.text}" for m in messages])
    
    # Add system prompt if provided
    if params.systemPrompt:
        prompt = f"{params.systemPrompt}\n\n{prompt}"
    
    # Generate response
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": params.temperature or 0.7,
            "max_output_tokens": params.maxTokens or 1000,
        }
    )
    
    return response.text

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
async def batch_operation(items: list) -> dict:
    """Process with progress."""
    for i, item in enumerate(items):
        await report_progress(i+1, len(items), f"Processing {item}")
        await process(item)
    return {"status": "complete"}

if __name__ == "__main__":
    mcp.run()
```

## Resources

- [FastMCP Client Docs](https://gofastmcp.com/clients)
- [OpenAPI Integration](https://gofastmcp.com/integrations/openapi)
- [Gemini SDK Integration](https://gofastmcp.com/integrations/gemini)
- [Tool Transformation](https://gofastmcp.com/patterns/tool-transformation)
- [MCP Protocol Spec](https://modelcontextprotocol.io)