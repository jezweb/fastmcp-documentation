# FastMCP Complete Development Guide

A comprehensive guide for building, deploying, and integrating FastMCP servers based on real-world experience and best practices.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Building Your First Server](#building-your-first-server)
4. [API Integration](#api-integration)
5. [Deployment](#deployment)
6. [Best Practices](#best-practices)
7. [Advanced Patterns](#advanced-patterns)
8. [Testing & Development](#testing--development)
9. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites
- Python 3.8+
- GitHub account (for FastMCP Cloud deployment)
- pip or uv package manager

### Installation
```bash
pip install fastmcp
# or
uv pip install fastmcp
```

### Minimal Server Example
```python
from fastmcp import FastMCP

mcp = FastMCP("My Server")

@mcp.tool
def hello(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    mcp.run()
```

## Core Concepts

### 1. Tools
Tools are functions that LLMs can call to perform actions:

```python
@mcp.tool
def calculate(operation: str, a: float, b: float) -> float:
    """Perform mathematical operations."""
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None
    }
    return operations.get(operation, lambda x, y: None)(a, b)
```

**Best Practices for Tools:**
- Clear, descriptive function names
- Comprehensive docstrings (LLMs read these!)
- Strong type hints
- Validate inputs
- Return structured data (dicts/lists)
- Handle errors gracefully

### 2. Resources
Resources expose static or dynamic data:

```python
@mcp.resource("data://config")
def get_config() -> dict:
    """Provide application configuration."""
    return {
        "version": "1.0.0",
        "features": ["auth", "api", "cache"]
    }
```

**Resource URI Patterns:**
- `data://` - Generic data
- `file://` - File resources
- `resource://` - General resources
- `info://` - Information/metadata

### 3. Resource Templates
Dynamic resources with parameters:

```python
@mcp.resource("user://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    """Get user profile by ID."""
    # Fetch from database/API
    return {"id": user_id, "name": "User Name"}
```

### 4. Prompts
Pre-configured prompts for LLMs:

```python
@mcp.prompt("analyze")
def analyze_prompt(topic: str) -> str:
    """Generate analysis prompt."""
    return f"""
    Analyze {topic} considering:
    1. Current state
    2. Challenges
    3. Opportunities
    4. Recommendations
    """
```

## Building Your First Server

### Step 1: Create Project Structure
```
my-mcp-server/
├── server.py          # Main server file
├── requirements.txt   # Dependencies
├── .env              # Environment variables (git-ignored)
├── .gitignore        # Git ignore file
└── README.md         # Documentation
```

### Step 2: Design Your Server
```python
from fastmcp import FastMCP
import os
from datetime import datetime

# Initialize with meaningful name
mcp = FastMCP(
    name="MyApp MCP Server",
    instructions="""
    This server provides tools for MyApp functionality.
    Use 'search' for data queries, 'process' for operations.
    """
)

# Environment configuration
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.example.com")

@mcp.tool
def search(query: str, limit: int = 10) -> dict:
    """Search for items matching query."""
    # Example implementation
    results = []
    # Your search logic here
    # results = database.search(query, limit)
    return {"query": query, "results": results, "count": len(results)}

@mcp.resource("info://status")
def server_status() -> dict:
    """Get server status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    mcp.run()
```

### Step 3: Add Dependencies
```txt
fastmcp>=0.3.0
httpx  # For API calls
python-dotenv  # For environment variables
```

## API Integration

### Method 1: Direct API Integration
```python
import httpx
from fastmcp import FastMCP

mcp = FastMCP("API Integration Server")

# Create HTTP client
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {API_TOKEN}"}
)

@mcp.tool
async def fetch_data(endpoint: str) -> dict:
    """Fetch data from API."""
    response = await client.get(endpoint)
    response.raise_for_status()
    return response.json()
```

### Method 2: OpenAPI/Swagger Integration
```python
import httpx
from fastmcp import FastMCP

# Load OpenAPI spec
spec = httpx.get("https://api.example.com/openapi.json").json()

# Create authenticated client
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {API_TOKEN}"},
    timeout=30.0
)

# Auto-generate MCP server from OpenAPI
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    name="API Server"
)

# Optionally add custom tools
@mcp.tool
def process_api_data(data: dict) -> dict:
    """Process data from API."""
    # Custom processing logic
    return {"processed": data}

if __name__ == "__main__":
    mcp.run(transport="http", port=8000)
```

### Method 3: FastAPI Integration
```python
from fastapi import FastAPI
from fastmcp import FastMCP

# Existing FastAPI app
app = FastAPI()

@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"id": item_id, "name": "Item"}

# Convert to MCP server
mcp = FastMCP.from_fastapi(app=app)

# Add authentication if needed
mcp_with_auth = FastMCP.from_fastapi(
    app=app,
    httpx_client_kwargs={
        "headers": {"Authorization": "Bearer token"}
    }
)
```

### Route Mapping for APIs
```python
from fastmcp.server.openapi import RouteMap, MCPType

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    route_maps=[
        # GET with parameters → Resource Templates
        RouteMap(
            methods=["GET"],
            pattern=r".*\{.*\}.*",
            mcp_type=MCPType.RESOURCE_TEMPLATE
        ),
        # GET without parameters → Resources
        RouteMap(
            methods=["GET"],
            mcp_type=MCPType.RESOURCE
        ),
        # POST/PUT/DELETE → Tools
        RouteMap(
            methods=["POST", "PUT", "DELETE"],
            mcp_type=MCPType.TOOL
        ),
    ]
)
```

## Deployment

### FastMCP Cloud Deployment

#### 1. Prepare Your Repository
```bash
git init
git add .
git commit -m "Initial MCP server"
gh repo create my-mcp-server --public
git push -u origin main
```

#### 2. Deploy on FastMCP Cloud
1. Visit [fastmcp.cloud](https://fastmcp.cloud)
2. Sign in with GitHub
3. Click "Create Project"
4. Select your repository
5. Configure:
   - **Server Name**: Your project name
   - **Entrypoint**: `server.py`
   - **Authentication**: Optional
   - **Environment Variables**: Add any needed

#### 3. Access Your Server
- URL: `https://your-project.fastmcp.app/mcp`
- Automatic deployment on push to main
- PR preview deployments

### Local Development
```bash
# Run with stdio transport (default)
python server.py

# Run with HTTP transport
python server.py --transport http --port 8000

# Run with FastMCP CLI
fastmcp run server.py

# Development mode with inspector
fastmcp dev server.py
```

### Client Configuration

#### Claude Desktop
```json
{
  "mcpServers": {
    "my-server": {
      "url": "https://your-project.fastmcp.app/mcp",
      "transport": "http"
    }
  }
}
```

#### Local Development
```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "API_KEY": "your-key"
      }
    }
  }
}
```

## Best Practices

### 1. Server Structure
```python
from fastmcp import FastMCP

def create_server() -> FastMCP:
    """Factory function for complex setup."""
    mcp = FastMCP("Server Name")
    
    # Configure server
    setup_tools(mcp)
    setup_resources(mcp)
    
    return mcp

def setup_tools(mcp: FastMCP):
    """Register all tools."""
    @mcp.tool
    def calculate(operation: str, a: float, b: float) -> float:
        """Perform basic math operations."""
        ops = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else None
        }
        return ops.get(operation, lambda x, y: None)(a, b)

def setup_resources(mcp: FastMCP):
    """Register all resources."""
    @mcp.resource("data://config")
    def get_config():
        """Provide application configuration."""
        return {
            "version": "1.0.0",
            "environment": "production",
            "features": ["auth", "api", "cache"]
        }

# Allow both direct run and factory
mcp = create_server()

if __name__ == "__main__":
    mcp.run()
```

### 2. Error Handling
```python
@mcp.tool
def safe_operation(param: str) -> dict:
    """Operation with error handling."""
    try:
        # Validate input
        if not param:
            return {"error": "Parameter required"}
        
        # Perform operation
        # Example: process the parameter
        result = {"processed": param.upper(), "length": len(param)}
        
        return {"success": True, "data": result}
    
    except ValidationError as e:
        return {"error": f"Validation failed: {e}"}
    except Exception as e:
        return {"error": f"Operation failed: {str(e)}"}
```

### 3. Environment Configuration
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    API_KEY = os.getenv("API_KEY", "")
    BASE_URL = os.getenv("BASE_URL", "https://api.example.com")
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    TIMEOUT = int(os.getenv("TIMEOUT", "30"))

# Use config
@mcp.tool
def api_call():
    if not Config.API_KEY:
        return {"error": "API key not configured"}
    # Make API call
```

### 4. Documentation
```python
@mcp.tool
def complex_tool(
    query: str,
    filters: dict = None,
    limit: int = 10
) -> dict:
    """
    Search with advanced filtering.
    
    Args:
        query: Search query string
        filters: Optional filters dict with keys:
            - category: Filter by category
            - date_from: Start date (ISO format)
            - date_to: End date (ISO format)
        limit: Maximum results (1-100)
    
    Returns:
        Dict with 'results' list and 'total' count
    
    Examples:
        >>> complex_tool("python", {"category": "tutorial"}, 5)
    """
    # Example implementation
    results = []
    
    # Mock data for example
    data = [
        {"title": "Python Tutorial", "category": "tutorial"},
        {"title": "Python Guide", "category": "guide"},
        {"title": "Java Tutorial", "category": "tutorial"}
    ]
    
    # Apply filters if provided
    filtered_data = data
    if filters and "category" in filters:
        filtered_data = [d for d in data if d.get("category") == filters["category"]]
    
    # Search in filtered data
    for item in filtered_data:
        if query.lower() in item.get("title", "").lower():
            results.append(item)
    
    # Apply limit
    results = results[:limit]
    
    return {"results": results, "total": len(results)}
```

## Advanced Patterns

### 1. Tool Transformation
```python
from fastmcp import Tool

# Original tool
@mcp.tool
def original_tool(x: int, y: int) -> int:
    return x + y

# Transform for better LLM usage
async def validate_positive(x: int, y: int) -> int:
    if x <= 0 or y <= 0:
        raise ValueError("Values must be positive")
    return await forward(x=x, y=y)

enhanced_tool = Tool.from_tool(
    original_tool,
    name="add_positive_numbers",
    description="Add two positive numbers",
    transform_fn=validate_positive
)

mcp.add_tool(enhanced_tool)
```

### 2. Server Composition
```python
# Sub-server for specific functionality
weather_mcp = FastMCP("Weather Service")

@weather_mcp.tool
def get_weather(city: str) -> dict:
    return {"city": city, "temp": 72}

# Main server
main_mcp = FastMCP("Main Server")

# Mount sub-server
main_mcp.mount(weather_mcp, prefix="weather")

# Now available as: weather_get_weather
```

### 3. Authentication
```python
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    issuer="https://auth.example.com",
    audience="my-mcp-server"
)

mcp = FastMCP("Protected Server", auth=auth)
```

### 4. Async Operations
```python
import asyncio
import httpx

@mcp.tool
async def async_operation(url: str) -> dict:
    """Perform async HTTP request."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

@mcp.tool
async def parallel_operations(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs in parallel."""
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

## Testing & Development

### 1. Testing with FastMCP Client
```python
import asyncio
from fastmcp import Client

async def test_server():
    async with Client("server.py") as client:
        # List tools
        tools = await client.list_tools()
        print(f"Tools: {[t.name for t in tools]}")
        
        # Call tool
        result = await client.call_tool("hello", {"name": "Test"})
        assert result.data == "Hello, Test!"
        
        # Read resource
        resource = await client.read_resource("data://config")
        print(f"Config: {resource}")

asyncio.run(test_server())
```

### 2. Development Workflow
```bash
# Install in editable mode
pip install -e .

# Run with inspector
fastmcp dev server.py

# Test specific functionality
python -m pytest tests/

# Check with linting
ruff check server.py
mypy server.py
```

### 3. Debugging
```python
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@mcp.tool
def debug_tool(param: str) -> dict:
    logger.debug(f"Received param: {param}")
    
    try:
        result = process(param)
        logger.info(f"Success: {result}")
        return result
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```python
# Wrong
from fastmcp import mcp  # No such import

# Correct
from fastmcp import FastMCP
mcp = FastMCP("Server Name")
```

#### 2. Async/Sync Mismatch
```python
# Wrong - mixing async/sync
@mcp.tool
def sync_tool():
    result = await async_function()  # Error!

# Correct - use async
@mcp.tool
async def async_tool():
    result = await async_function()
    return result
```

#### 3. Environment Variables
```python
# Always provide defaults
API_KEY = os.getenv("API_KEY", "")

# Check before use
if not API_KEY:
    raise ValueError("API_KEY environment variable required")
```

#### 4. Resource URIs
```python
# Wrong
@mcp.resource("config")  # Missing scheme

# Correct
@mcp.resource("data://config")  # Has scheme
```

### Debugging Tips

1. **Enable Debug Logging:**
```python
mcp.run(log_level="debug")
```

2. **Test Locally First:**
```bash
fastmcp dev server.py
```

3. **Check Server Status:**
```python
@mcp.resource("info://health")
def health_check() -> dict:
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "api": check_api_connection(),
            "database": check_db_connection()
        }
    }
```

4. **Validate OpenAPI Integration:**
```python
# Test OpenAPI parsing
routes = parse_openapi_to_http_routes(spec)
print(f"Found {len(routes)} routes")

# Test with client
async with Client(mcp) as client:
    tools = await client.list_tools()
    print(f"Generated {len(tools)} tools")
```

## Summary

FastMCP provides a powerful, flexible framework for building MCP servers that can:
- Expose tools, resources, and prompts to LLMs
- Integrate with any API via OpenAPI/Swagger
- Deploy instantly to FastMCP Cloud
- Scale from simple scripts to complex applications

Key takeaways:
1. Start simple, iterate based on needs
2. Use type hints and docstrings extensively
3. Handle errors gracefully
4. Test locally before deploying
5. Leverage OpenAPI for existing APIs
6. Use environment variables for configuration
7. Follow MCP patterns for better LLM integration

For more examples and updates, visit:
- [FastMCP Documentation](https://docs.fastmcp.com)
- [GitHub Repository](https://github.com/jlowin/fastmcp)
- [FastMCP Cloud](https://fastmcp.cloud)