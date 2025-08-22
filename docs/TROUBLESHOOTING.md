# FastMCP Troubleshooting Guide

## Common Issues and Solutions

### Installation & Setup Issues

#### FastMCP Installation Fails
```bash
# Error: ModuleNotFoundError: No module named 'fastmcp'
# Solution: Install with correct version
pip install "fastmcp>=0.3.0"

# Error: Version conflicts
# Solution: Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
pip install fastmcp
```

#### Server Won't Start
```python
# Error: ImportError: cannot import name 'mcp' from 'fastmcp'
# Wrong:
from fastmcp import mcp

# Correct:
from fastmcp import FastMCP
mcp = FastMCP("Server Name")

# Error: Server exits immediately
# Solution: Ensure run() is called
if __name__ == "__main__":
    mcp.run()
```

#### Environment Variables Not Loading
```python
# Problem: os.getenv returns None
# Solution 1: Use python-dotenv
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# Solution 2: Provide defaults
API_KEY = os.getenv("API_KEY", "")
if not API_KEY:
    print("Warning: API_KEY not set")

# Solution 3: Check .env location
# .env must be in same directory as script or specify path
load_dotenv(dotenv_path="/path/to/.env")
```

### Client Handler Issues

#### Elicitation Handler Not Working
```python
# Problem: User input not being requested
# Check 1: Handler registered correctly
async def handle_elicitation(message: str, response_type: type, context: dict):
    # Implementation
    return input(f"{message}: ")

client = Client("server.py", elicitation_handler=handle_elicitation)

# Check 2: Server using elicit correctly
from fastmcp import elicit

@mcp.tool
async def needs_input():
    # Must use await with elicit
    user_input = await elicit("Enter value:", str)
    return user_input

# Debug: Add logging
async def handle_elicitation(message: str, response_type: type, context: dict):
    print(f"[ELICIT] Message: {message}, Type: {response_type}")
    result = input(f"{message}: ")
    print(f"[ELICIT] Got: {result}")
    return result
```

#### Progress Handler Not Updating
```python
# Problem: Progress not showing
# Check 1: Using report_progress correctly
from fastmcp import report_progress

@mcp.tool
async def long_task():
    for i in range(100):
        # Must await report_progress
        await report_progress(i, 100, f"Processing {i}")
        await asyncio.sleep(0.1)

# Check 2: Handler implementation
async def handle_progress(progress: float, total: float | None, message: str | None):
    if total:
        pct = (progress / total) * 100
        # Use \r for same line update
        print(f"\r[{pct:.1f}%] {message}", end="", flush=True)
    else:
        print(f"Progress: {progress} - {message}")

# Debug: Verify handler called
async def handle_progress(progress, total, message):
    print(f"[PROGRESS] {progress}/{total}: {message}")
```

#### Sampling Handler Errors
```python
# Problem: LLM requests failing
# Check 1: Handler returns correct format
async def handle_sampling(messages, params, context):
    print(f"[SAMPLE] Messages: {messages}")
    print(f"[SAMPLE] Params: {params}")
    
    # Must return dict with 'content' key
    return {
        "content": "Generated response",
        "model": params.get("model", "default"),
        "usage": {"tokens": 100}
    }

# Check 2: Server sampling request
from fastmcp import sample

@mcp.tool
async def ai_tool():
    # Correct message format
    messages = [
        {"role": "user", "content": "Generate text"}
    ]
    
    result = await sample(
        messages=messages,
        model="gpt-4",
        temperature=0.7
    )
    
    return result["content"]

# Debug: Mock handler for testing
async def mock_sampling(messages, params, context):
    return {"content": f"Mock response for: {messages[-1]['content']}"}
```

### Resource & Template Issues

#### Resource Not Found
```python
# Error: Resource 'config' not found
# Wrong: Missing scheme
@mcp.resource("config")

# Correct: Include scheme
@mcp.resource("data://config")
@mcp.resource("file://config.json")
@mcp.resource("api://status")

# Debug: List all resources
async def test():
    async with Client("server.py") as client:
        resources = await client.list_resources()
        for r in resources:
            print(f"- {r.uri}: {r.description}")
```

#### Resource Template Parameters Not Working
```python
# Problem: Template parameters not passed
# Wrong: Missing parameter in function
@mcp.resource("user://{user_id}/profile")
def get_user():  # Missing user_id!
    return {"error": "No user_id"}

# Correct: Parameter must match template
@mcp.resource("user://{user_id}/profile")
def get_user(user_id: str):
    return {"id": user_id, "name": f"User {user_id}"}

# Multiple parameters
@mcp.resource("api://{version}/users/{user_id}")
def get_versioned_user(version: str, user_id: str):
    return {"version": version, "user_id": user_id}

# Debug: Test with client
async def test():
    async with Client("server.py") as client:
        # Correct URI format
        data = await client.read_resource("user://123/profile")
        print(data)
```

### OpenAPI Integration Issues

#### OpenAPI Spec Loading Fails
```python
# Problem: Can't load OpenAPI spec
# Solution 1: Check URL accessibility
import httpx

try:
    response = httpx.get("https://api.example.com/openapi.json")
    response.raise_for_status()
    spec = response.json()
except httpx.HTTPError as e:
    print(f"Failed to load spec: {e}")

# Solution 2: Handle authentication
client = httpx.AsyncClient(
    headers={"Authorization": f"Bearer {token}"}
)
spec_response = await client.get("/openapi.json")
spec = spec_response.json()

# Solution 3: Local spec file
import json
with open("openapi.json") as f:
    spec = json.load(f)

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    name="API Server"
)
```

#### Route Mapping Incorrect
```python
# Problem: Routes not mapping to correct MCP types
# Debug: Print route analysis
from fastmcp.server.openapi import analyze_routes

routes = analyze_routes(spec)
for path, methods in routes.items():
    print(f"{path}:")
    for method, details in methods.items():
        print(f"  {method}: {details}")

# Custom mapping
from fastmcp.server.openapi import RouteMap, MCPType

route_maps = [
    # Be specific with patterns
    RouteMap(
        methods=["GET"],
        pattern=r"^/users/\{user_id\}$",  # Exact match
        mcp_type=MCPType.RESOURCE_TEMPLATE
    ),
    RouteMap(
        methods=["GET"],
        pattern=r"^/users$",  # List endpoint
        mcp_type=MCPType.RESOURCE
    ),
]

mcp = FastMCP.from_openapi(
    openapi_spec=spec,
    client=client,
    route_maps=route_maps
)
```

#### API Client Errors
```python
# Problem: API calls failing
# Solution 1: Check base URL
client = httpx.AsyncClient(
    base_url="https://api.example.com",  # No trailing slash
    timeout=30.0
)

# Solution 2: Handle rate limiting
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def api_call():
    response = await client.get("/endpoint")
    response.raise_for_status()
    return response.json()

# Solution 3: Debug requests
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use event hooks
async def log_request(request):
    print(f"Request: {request.method} {request.url}")

async def log_response(response):
    print(f"Response: {response.status_code}")

client = httpx.AsyncClient(
    event_hooks={
        "request": [log_request],
        "response": [log_response]
    }
)
```

### Tool Transformation Issues

#### Transform Function Not Working
```python
# Problem: Transformation not applied
# Check 1: Using forward correctly
from fastmcp import Tool, forward

async def validate_transform(x: int, y: int) -> int:
    # Validate
    if x < 0 or y < 0:
        raise ValueError("Must be positive")
    
    # Must use forward to call original
    return await forward(x=x, y=y)

# Check 2: Correct tool creation
original = mcp.get_tool("add")
enhanced = Tool.from_tool(
    original,
    transform_fn=validate_transform
)

# Register enhanced tool
mcp.add_tool(enhanced)

# Debug: Test both versions
async def test():
    # Original
    result1 = await original.fn(x=5, y=3)
    print(f"Original: {result1}")
    
    # Enhanced
    try:
        result2 = await enhanced.fn(x=-1, y=3)
    except ValueError as e:
        print(f"Enhanced caught: {e}")
```

#### Decorator Chain Issues
```python
# Problem: Multiple decorators conflict
# Solution: Order matters
async def cache_transform(key: str):
    # Check cache first
    if cached := cache.get(key):
        return cached
    # Call original
    result = await forward(key=key)
    cache[key] = result
    return result

async def validate_transform(key: str):
    if not key:
        raise ValueError("Key required")
    return await forward(key=key)

# Apply in correct order (validation first, then cache)
tool = Tool.from_tool(original, transform_fn=validate_transform)
tool = Tool.from_tool(tool, transform_fn=cache_transform)
```

### Authentication Issues

#### JWT/Bearer Token Failures
```python
# Problem: Authentication rejected
# Debug 1: Check token format
import jwt

try:
    # Decode without verification to inspect
    payload = jwt.decode(token, options={"verify_signature": False})
    print(f"Token payload: {payload}")
    print(f"Expires: {payload.get('exp')}")
except jwt.DecodeError as e:
    print(f"Invalid token format: {e}")

# Debug 2: Verify JWKS
from fastmcp.server.auth import BearerAuthProvider

auth = BearerAuthProvider(
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    issuer="https://auth.example.com",
    audience="my-server",
    # Add debug logging
    debug=True
)

# Debug 3: Manual token validation
import httpx
from jwt import PyJWKClient

jwks_client = PyJWKClient("https://auth.example.com/.well-known/jwks.json")
signing_key = jwks_client.get_signing_key_from_jwt(token)

payload = jwt.decode(
    token,
    signing_key.key,
    algorithms=["RS256"],
    audience="my-server",
    issuer="https://auth.example.com"
)
print(f"Valid token: {payload}")
```

#### OAuth2 Flow Issues
```python
# Problem: OAuth2 callback not working
# Solution 1: Check redirect URI
from fastmcp.server.auth import OAuth2Provider

auth = OAuth2Provider(
    client_id="your-client-id",
    client_secret="your-secret",
    auth_url="https://auth.example.com/authorize",
    token_url="https://auth.example.com/token",
    redirect_uri="http://localhost:8000/callback",  # Must match exactly
    scopes=["read", "write"]
)

# Solution 2: Handle state parameter
@app.get("/callback")
async def oauth_callback(code: str, state: str):
    # Verify state to prevent CSRF
    if not verify_state(state):
        raise ValueError("Invalid state")
    
    # Exchange code for token
    token = await auth.exchange_code(code)
    return token

# Debug: Log OAuth flow
import logging
logging.getLogger("fastmcp.auth").setLevel(logging.DEBUG)
```

### Performance Issues

#### Server Slow to Respond
```python
# Problem: Tools taking too long
# Solution 1: Use async for I/O
# Wrong: Blocking I/O
@mcp.tool
def slow_tool():
    response = requests.get("https://api.example.com")  # Blocks!
    return response.json()

# Correct: Async I/O
@mcp.tool
async def fast_tool():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()

# Solution 2: Connection pooling
# Create client once, reuse
client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10
    )
)

@mcp.tool
async def efficient_tool(endpoint: str):
    response = await client.get(endpoint)
    return response.json()

# Solution 3: Caching
from functools import lru_cache
import time

cache = {}
cache_ttl = 300  # 5 minutes

@mcp.tool
async def cached_tool(key: str):
    # Check cache
    if key in cache:
        cached_data, timestamp = cache[key]
        if time.time() - timestamp < cache_ttl:
            return cached_data
    
    # Fetch fresh data
    data = await fetch_data(key)
    cache[key] = (data, time.time())
    return data
```

#### Memory Usage High
```python
# Problem: Server using too much memory
# Solution 1: Limit cache size
from functools import lru_cache

@lru_cache(maxsize=100)  # Limit to 100 entries
def expensive_computation(param: str):
    return compute(param)

# Solution 2: Stream large responses
@mcp.tool
async def stream_large_file(file_path: str):
    # Don't load entire file
    async def generate():
        async with aiofiles.open(file_path, 'r') as f:
            async for line in f:
                yield line
    
    return StreamingResponse(generate())

# Solution 3: Clear unused objects
import gc

@mcp.tool
async def memory_intensive_task():
    large_data = process_data()
    result = extract_summary(large_data)
    
    # Clear large data
    del large_data
    gc.collect()
    
    return result
```

### Deployment Issues

#### FastMCP Cloud Deployment Fails
```bash
# Problem: Deployment errors
# Check 1: Requirements file
# Ensure all dependencies listed
pip freeze > requirements.txt

# Check 2: Python version
# Specify in runtime.txt if needed
echo "python-3.11" > runtime.txt

# Check 3: Entry point
# Ensure server.py has if __name__ == "__main__":
if __name__ == "__main__":
    mcp.run()

# Check 4: Environment variables
# Set in FastMCP Cloud dashboard, not in code
# Don't commit .env files
```

#### Client Can't Connect
```json
// Problem: Claude Desktop can't connect
// Check 1: Correct URL format
{
  "mcpServers": {
    "my-server": {
      // For deployed
      "url": "https://project.fastmcp.app/mcp",  // Note /mcp suffix
      "transport": "http",
      
      // For local
      "command": "python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}

// Check 2: Authentication headers if needed
{
  "mcpServers": {
    "my-server": {
      "url": "https://project.fastmcp.app/mcp",
      "transport": "http",
      "headers": {
        "Authorization": "Bearer your-token"
      }
    }
  }
}
```

### Debugging Techniques

#### Enable Debug Logging
```python
# Method 1: In code
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Method 2: Via run
mcp.run(log_level="debug")

# Method 3: Environment variable
os.environ["FASTMCP_LOG_LEVEL"] = "DEBUG"

# Method 4: Specific loggers
logging.getLogger("fastmcp").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
```

#### Add Health Check
```python
@mcp.tool
def health_check() -> dict:
    """System health check."""
    checks = {}
    
    # Check API connectivity
    try:
        response = httpx.get("https://api.example.com/health")
        checks["api"] = response.status_code == 200
    except:
        checks["api"] = False
    
    # Check environment
    checks["env_vars"] = bool(os.getenv("API_KEY"))
    
    # Check dependencies
    checks["dependencies"] = True
    try:
        import required_module
    except ImportError:
        checks["dependencies"] = False
    
    return {
        "status": "healthy" if all(checks.values()) else "unhealthy",
        "checks": checks,
        "timestamp": datetime.now().isoformat()
    }
```

#### Test Harness
```python
# test_server.py
import asyncio
from fastmcp import Client

async def test_server():
    """Test all server functionality."""
    async with Client("server.py") as client:
        print("Testing server...")
        
        # Test tools
        tools = await client.list_tools()
        print(f"Tools: {len(tools)}")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description}")
        
        # Test each tool
        for tool in tools:
            try:
                if tool.name == "health_check":
                    result = await client.call_tool(tool.name, {})
                    print(f"  ✓ {tool.name}: {result}")
            except Exception as e:
                print(f"  ✗ {tool.name}: {e}")
        
        # Test resources
        resources = await client.list_resources()
        print(f"Resources: {len(resources)}")
        for resource in resources:
            try:
                data = await client.read_resource(resource.uri)
                print(f"  ✓ {resource.uri}")
            except Exception as e:
                print(f"  ✗ {resource.uri}: {e}")

if __name__ == "__main__":
    asyncio.run(test_server())
```

#### Interactive Debugging
```python
# debug_server.py
import asyncio
from fastmcp import Client
import IPython

async def debug_session():
    """Start interactive debug session."""
    client = Client("server.py")
    await client.start()
    
    # Start IPython with client available
    IPython.embed(header="FastMCP Debug Session\nClient available as 'client'")
    
    await client.stop()

if __name__ == "__main__":
    asyncio.run(debug_session())
```

## Import and Module Issues

### Circular Import Error
```python
# Error: cannot import name 'X' from partially initialized module
# This occurs when modules import from each other in a circle

# ❌ WRONG: Circular dependency
# shared/__init__.py
from .monitoring import HealthCheck
def get_api_client():
    return APIClient()

# shared/monitoring.py
from . import get_api_client  # Creates circle!

# ✅ CORRECT: Direct import
# shared/monitoring.py
from .api_client import APIClient
client = APIClient()  # Create directly

# Alternative: Lazy import
# shared/monitoring.py
def get_client():
    from .api_client import APIClient
    return APIClient()
```

### Module Architecture Best Practices
```python
# ❌ AVOID: Factory functions in __init__.py
# shared/__init__.py
_client = None
def get_api_client():
    global _client
    if not _client:
        from .api_client import APIClient  # Can cause import loops
        _client = APIClient()
    return _client

# ✅ PREFER: Direct class imports
# shared/__init__.py
from .api_client import APIClient
from .cache import CacheManager
# Let consuming code create instances

# Usage in other modules
from shared.api_client import APIClient
client = APIClient()
```

### Python Version Compatibility
```python
# Cloud environments may use newer Python versions
# Handle deprecation warnings proactively

# ❌ Deprecated (Python 3.12+):
from datetime import datetime
timestamp = datetime.utcnow()

# ✅ Future-proof:
from datetime import datetime, timezone
timestamp = datetime.now(timezone.utc)

# ❌ Deprecated async pattern:
@asyncio.coroutine
def old_async():
    yield from asyncio.sleep(1)

# ✅ Modern async:
async def modern_async():
    await asyncio.sleep(1)
```

### Import Path Issues in Cloud
```python
# Cloud deployment may have different sys.path

# ❌ Relative imports that break:
from src.shared import config  # Works locally, fails in cloud

# ✅ Package-relative imports:
from .shared import config  # When in same package
from shared import config   # When shared is in sys.path

# Debug import issues:
import sys
print("Python path:", sys.path)
print("Current dir:", os.getcwd())

# Fix: Ensure proper package structure
project/
├── src/
│   ├── __init__.py  # Makes src a package
│   ├── server.py
│   └── shared/
│       ├── __init__.py
│       └── config.py
```

## Common Error Messages

### FastMCP Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: fastmcp` | Not installed | `pip install fastmcp` |
| `ImportError: cannot import name 'mcp'` | Wrong import | Use `from fastmcp import FastMCP` |
| `RuntimeError: No event loop` | Sync/async mismatch | Use `async def` for async operations |
| `ValueError: Invalid resource URI` | Missing scheme | Add scheme: `data://`, `file://`, etc. |
| `KeyError: forward` | Transform without forward | Import: `from fastmcp import forward` |
| `TimeoutError` | Slow operation | Increase timeout or optimize |
| `ConnectionError` | Can't reach API | Check network, URL, auth |

### Client Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Tool not found` | Tool not registered | Check `@mcp.tool` decorator |
| `Resource not found` | Wrong URI | Verify URI format and scheme |
| `Invalid parameters` | Type mismatch | Check parameter types |
| `Handler not called` | Not registered | Pass handler to Client init |
| `Elicitation timeout` | No user response | Implement timeout handling |

## Getting Help

### Resources
- [FastMCP Documentation](https://docs.fastmcp.com)
- [GitHub Issues](https://github.com/jlowin/fastmcp/issues)
- [Discord Community](https://discord.gg/fastmcp)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/fastmcp)

### Debug Checklist
1. ✓ Check FastMCP version: `pip show fastmcp`
2. ✓ Enable debug logging
3. ✓ Test with simple tool first
4. ✓ Verify environment variables
5. ✓ Check network connectivity
6. ✓ Review error messages carefully
7. ✓ Test with FastMCP client
8. ✓ Check GitHub issues for similar problems
9. ✓ Create minimal reproduction case
10. ✓ Ask for help with details

### Reporting Issues

When reporting issues, include:
- FastMCP version
- Python version
- Operating system
- Minimal code to reproduce
- Full error message
- What you expected vs what happened