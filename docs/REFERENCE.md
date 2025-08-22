# FastMCP Quick Reference Guide

## CLI Commands

### Running Servers
```bash
# Run with default STDIO transport
python server.py
fastmcp run server.py

# Run with HTTP transport
python server.py --transport http --port 8000
fastmcp run server.py --transport http

# Development mode with inspector
fastmcp dev server.py

# With specific Python version
fastmcp run server.py --python 3.11

# With dependencies
fastmcp dev server.py --with pandas --with httpx
```

### Installation Commands
```bash
# Install in Claude Desktop
fastmcp install claude-desktop server.py

# Install in Claude Code
fastmcp install claude-code server.py

# Install in Cursor
fastmcp install cursor server.py

# Generate MCP JSON config
fastmcp install mcp-json server.py

# With environment variables
fastmcp install claude-desktop server.py --env API_KEY=secret

# With environment file
fastmcp install cursor server.py --env-file .env

# Copy config to clipboard
fastmcp install mcp-json server.py --copy
```

### Other Commands
```bash
# Inspect server
fastmcp inspect server.py

# Check version
fastmcp version
```

## Core Patterns

### Basic Server
```python
from fastmcp import FastMCP

mcp = FastMCP("Server Name")

@mcp.tool
def my_tool(param: str) -> str:
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run()
```

### Factory Pattern
```python
def create_server() -> FastMCP:
    mcp = FastMCP("Server")
    # Complex setup
    return mcp

mcp = create_server()
```

### Environment Variables
```python
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.example.com")
```

## API Integration Patterns

### OpenAPI Integration
```python
import httpx
from fastmcp import FastMCP

# Load spec
spec = httpx.get("https://api.example.com/openapi.json").json()

# Create client
client = httpx.AsyncClient(
    base_url="https://api.example.com",
    headers={"Authorization": f"Bearer {token}"}
)

# Generate server
mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    name="API Server"
)
```

### FastAPI Integration
```python
from fastapi import FastAPI
from fastmcp import FastMCP

app = FastAPI()
mcp = FastMCP.from_fastapi(app=app)
```

### Route Mapping
```python
from fastmcp.server.openapi import RouteMap, MCPType

route_maps = [
    # GET with params → ResourceTemplate
    RouteMap(
        methods=["GET"],
        pattern=r".*\{.*\}.*",
        mcp_type=MCPType.RESOURCE_TEMPLATE
    ),
    # GET → Resource
    RouteMap(
        methods=["GET"],
        mcp_type=MCPType.RESOURCE
    ),
    # POST/PUT/DELETE → Tool
    RouteMap(
        methods=["POST", "PUT", "DELETE"],
        mcp_type=MCPType.TOOL
    ),
]
```

## Client Handlers

### Elicitation Handler
```python
from fastmcp import Client

async def handle_elicitation(message: str, response_type: type, context: dict):
    """Handle interactive input requests."""
    return input(f"{message}: ")

client = Client("server.py", elicitation_handler=handle_elicitation)
```

### Progress Handler
```python
async def handle_progress(progress: float, total: float | None, message: str | None):
    """Track operation progress."""
    if total:
        pct = (progress / total) * 100
        print(f"[{pct:.1f}%] {message}")

client = Client("server.py", progress_handler=handle_progress)
```

### Sampling Handler
```python
async def handle_sampling(messages, params, context):
    """Handle LLM requests from server."""
    # Integrate with your preferred LLM
    return llm.generate(messages, **params)

client = Client("server.py", sampling_handler=handle_sampling)
```

## Component Types

### Tools
```python
@mcp.tool
def tool_name(param: str, optional: int = 10) -> dict:
    """Tool description for LLM."""
    return {"result": param}

# Async tool
@mcp.tool
async def async_tool(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Resources
```python
@mcp.resource("data://config")
def get_config() -> dict:
    """Static resource."""
    return {"key": "value"}

# Async resource
@mcp.resource("api://status")
async def api_status() -> dict:
    # Check API status
    return {"status": "healthy"}
```

### Resource Templates
```python
@mcp.resource("user://{user_id}/profile")
def get_user(user_id: str) -> dict:
    """Dynamic resource with parameters."""
    return {"id": user_id}
```

### Prompts
```python
@mcp.prompt("analyze")
def analyze_prompt(topic: str) -> str:
    """Generate analysis prompt."""
    return f"Analyze {topic}..."
```

## Advanced Patterns

### Tool Transformation
```python
from fastmcp import Tool, forward

async def validate(x: int, y: int) -> int:
    if x < 0 or y < 0:
        raise ValueError("Must be positive")
    return await forward(x=x, y=y)

enhanced = Tool.from_tool(
    original_tool,
    transform_fn=validate
)
```

### Server Composition
```python
sub_server = FastMCP("Sub")
main_server = FastMCP("Main")

# Mount sub-server
main_server.mount(sub_server, prefix="sub")
```

### Authentication
```python
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    issuer="https://auth.example.com",
    audience="my-server"
)

mcp = FastMCP("Protected", auth=auth)
```

## Testing

### With FastMCP Client
```python
import asyncio
from fastmcp import Client

async def test():
    async with Client("server.py") as client:
        # List tools
        tools = await client.list_tools()
        
        # Call tool
        result = await client.call_tool("tool_name", {"param": "value"})
        
        # Read resource
        data = await client.read_resource("data://config")

asyncio.run(test())
```

### With HTTP Client
```python
import httpx

# For HTTP transport servers
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/mcp/tools/tool_name",
        json={"param": "value"}
    )
```

## Deployment Checklist

### FastMCP Cloud
- [ ] Create GitHub repository
- [ ] Push code to main branch
- [ ] Sign in to fastmcp.cloud with GitHub
- [ ] Create project and select repo
- [ ] Configure server name and entrypoint
- [ ] Add environment variables if needed
- [ ] Deploy

### Local Development
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create `.env` file with secrets
- [ ] Test with: `fastmcp dev server.py`
- [ ] Run locally: `python server.py`

### Client Configuration
```json
// Claude Desktop / Cursor
{
  "mcpServers": {
    "my-server": {
      // For local
      "command": "python",
      "args": ["path/to/server.py"],
      
      // For deployed
      "url": "https://project.fastmcp.app/mcp",
      "transport": "http"
    }
  }
}
```

## Common Issues & Solutions

### Import Error
```python
# Wrong
from fastmcp import mcp

# Correct
from fastmcp import FastMCP
mcp = FastMCP("Name")
```

### Async/Sync Mismatch
```python
# Wrong
@mcp.tool
def sync_tool():
    result = await async_func()  # Error!

# Correct
@mcp.tool
async def async_tool():
    result = await async_func()
    return result
```

### Missing Resource Scheme
```python
# Wrong
@mcp.resource("config")

# Correct
@mcp.resource("data://config")
```

### Environment Variables
```python
# Always provide defaults
API_KEY = os.getenv("API_KEY", "")

# Check before use
if not API_KEY:
    return {"error": "API_KEY required"}
```

## Error Handling

### Tool Error Handling
```python
@mcp.tool
def safe_tool(param: str) -> dict:
    try:
        # Validate
        if not param:
            return {"error": "Parameter required"}
        
        # Process
        result = process(param)
        return {"success": True, "data": result}
        
    except ValidationError as e:
        return {"error": f"Validation: {e}"}
    except Exception as e:
        return {"error": str(e)}
```

### Async Error Handling
```python
@mcp.tool
async def async_safe_tool(url: str) -> dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}
```

## Performance Tips

### Use Async for I/O
```python
# Good - async for network calls
@mcp.tool
async def fetch_data(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Batch Operations
```python
@mcp.tool
async def batch_fetch(urls: List[str]):
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_operation(param: str):
    # Cached result
    return compute(param)

@mcp.tool
def cached_tool(param: str):
    return expensive_operation(param)
```

## Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in run
mcp.run(log_level="debug")
```

### Health Check Tool
```python
@mcp.tool
def health_check() -> dict:
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "api": check_api(),
            "database": check_db()
        }
    }
```

### Test Mode
```python
if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        # Run tests
        asyncio.run(test_server())
    else:
        mcp.run()
```

## Resources

- [FastMCP Documentation](https://docs.fastmcp.com)
- [GitHub Repository](https://github.com/jlowin/fastmcp)
- [FastMCP Cloud](https://fastmcp.cloud)
- [MCP Protocol](https://modelcontextprotocol.io)
- [Examples](https://github.com/jlowin/fastmcp/tree/main/examples)