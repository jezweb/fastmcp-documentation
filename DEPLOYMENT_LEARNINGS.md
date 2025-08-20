# FastMCP Cloud Deployment Learnings

This document captures critical learnings from deploying production MCP servers to FastMCP Cloud, based on real-world experience.

## Table of Contents
1. [Critical Requirements](#critical-requirements)
2. [Server Object Exposure](#server-object-exposure)
3. [Dependency Management](#dependency-management)
4. [Repository Structure](#repository-structure)
5. [Environment Variables](#environment-variables)
6. [Resource Configuration](#resource-configuration)
7. [Testing Strategy](#testing-strategy)
8. [Common Pitfalls](#common-pitfalls)

## Critical Requirements

### Must-Have for FastMCP Cloud

1. **Module-level server object** named `mcp`, `server`, or `app`
2. **PyPI-only dependencies** in requirements.txt
3. **Public or accessible GitHub repository**
4. **Proper entry point** file structure
5. **Environment variables** for configuration

## Server Object Exposure

### The Golden Rule
FastMCP Cloud's runtime looks for a module-level variable. This is **non-negotiable**.

### ✅ CORRECT Pattern
```python
# server.py
from fastmcp import FastMCP

# MUST be at module level
mcp = FastMCP(
    name="my-server",
    version="1.0.0"
)

# Tools registered at module level
@mcp.tool()
async def my_tool(param: str) -> str:
    """Tool description."""
    return f"Result: {param}"

# Optional: main() for local execution
def main():
    import asyncio
    asyncio.run(mcp.run())

if __name__ == "__main__":
    main()
```

### ❌ INCORRECT Patterns

**Pattern 1: Function Wrapper (FAILS)**
```python
# This will cause "No server object found" error
def create_server():
    mcp = FastMCP(name="my-server")
    
    @mcp.tool()
    async def my_tool():
        pass
    
    return mcp

# Even if you do this, it won't work:
if __name__ == "__main__":
    server = create_server()  # Too late!
```

**Pattern 2: Class Wrapper (FAILS)**
```python
# This also fails
class MyServer:
    def __init__(self):
        self.mcp = FastMCP(name="my-server")
    
    def setup(self):
        @self.mcp.tool()
        async def my_tool():
            pass

# Won't work
server_instance = MyServer()
```

### Why This Matters
FastMCP Cloud imports your module and looks for the server object immediately. It doesn't execute your `main()` function or any initialization code.

## Dependency Management

### PyPI-Only Rule
FastMCP Cloud can only install packages from PyPI. No local packages, no git URLs, no editable installs.

### ❌ What Doesn't Work
```txt
# requirements.txt
-e ../shared-utils        # Editable local package
-e .                      # Current directory as editable
git+https://github.com/...  # Git URLs
../local-package          # Relative paths
file:///path/to/package   # File URLs
```

### ✅ What Works
```txt
# requirements.txt
fastmcp>=0.3.0
httpx>=0.24.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

### Handling Shared Code

**Option 1: Embed in Each Server (Recommended)**
```
project/
├── server-1/
│   ├── src/
│   │   ├── server.py
│   │   └── shared/      # Copied shared code
│   │       ├── __init__.py
│   │       ├── config.py
│   │       └── utils.py
│   └── requirements.txt
├── server-2/
│   ├── src/
│   │   ├── server.py
│   │   └── shared/      # Same shared code
│   │       └── ...
│   └── requirements.txt
```

**Option 2: Publish to PyPI**
```txt
# If you have reusable utilities
# 1. Publish to PyPI as a package
# 2. Reference in requirements.txt
my-shared-utils>=1.0.0
```

## Repository Structure

### Monorepo for Multiple Servers

**Recommended Structure:**
```
my-mcp-servers/
├── README.md
├── .gitignore
├── .env.example
│
├── server-ops/
│   ├── src/
│   │   └── server.py     # Entry point
│   └── requirements.txt
│
├── server-customer/
│   ├── src/
│   │   └── server.py     # Entry point
│   └── requirements.txt
│
└── server-finance/
    ├── src/
    │   └── server.py     # Entry point
    └── requirements.txt
```

### Deployment Configuration
For each server in FastMCP Cloud:
- **Repository**: `username/my-mcp-servers`
- **Branch**: `main` or `master`
- **Entrypoint**: `server-ops/src/server.py`
- **Requirements**: `server-ops/requirements.txt`

## Environment Variables

### Best Practices

1. **Always provide defaults**
```python
import os
from dotenv import load_dotenv

load_dotenv()

# Good: Has defaults
API_URL = os.getenv("API_URL", "https://api.example.com")
TIMEOUT = int(os.getenv("TIMEOUT", "30"))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Bad: Will crash if not set
API_KEY = os.environ["API_KEY"]  # KeyError if missing
```

2. **Validate on startup**
```python
class Config:
    API_KEY = os.getenv("API_KEY")
    API_URL = os.getenv("API_URL", "https://api.example.com")
    
    @classmethod
    def validate(cls):
        if not cls.API_KEY:
            raise ValueError("API_KEY environment variable is required")
        return True

# At module level
Config.validate()
```

3. **Document in .env.example**
```env
# Required
API_KEY=your-api-key-here
COMPANY_ID=12345

# Optional (defaults shown)
API_URL=https://api.example.com
LOG_LEVEL=INFO
CACHE_TTL=300
```

## Resource Configuration

### Recommended Settings

**Build Resources:**
- **Default**: 2 vCPU / 4GB RAM
- **Large servers**: 4 vCPU / 8GB RAM
- **Timeout**: 10 minutes max

**Runtime Resources:**
- **Start small**: 1 vCPU / 2GB RAM
- **Scale up if**:
  - Response time > 2 seconds
  - Memory usage > 80%
  - Frequent timeouts

### Cost Optimization
```python
# Use caching to reduce resource usage
from functools import lru_cache
import time

class Cache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
    
    def get(self, key):
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                return self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = value
        self.timestamps[key] = time.time()

# Global cache instance
cache = Cache(ttl=300)  # 5 minutes
```

## Testing Strategy

### Pre-Deployment Testing

1. **Local Module Import Test**
```python
# test_import.py
import sys
import importlib.util

# Test that server can be imported
spec = importlib.util.spec_from_file_location("server", "src/server.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Check for server object
if hasattr(module, 'mcp'):
    print("✅ Found 'mcp' object")
elif hasattr(module, 'server'):
    print("✅ Found 'server' object")
elif hasattr(module, 'app'):
    print("✅ Found 'app' object")
else:
    print("❌ No server object found!")
    sys.exit(1)
```

2. **Dependency Test**
```bash
# Create fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # or test_env\Scripts\activate on Windows

# Install only from requirements.txt
pip install -r requirements.txt

# Try to run
python src/server.py
```

3. **Environment Variable Test**
```python
# test_env.py
import os
from dotenv import load_dotenv

load_dotenv()

required = ["API_KEY", "API_URL", "COMPANY_ID"]
missing = []

for var in required:
    if not os.getenv(var):
        missing.append(var)

if missing:
    print(f"❌ Missing required env vars: {missing}")
else:
    print("✅ All required env vars present")
```

### Post-Deployment Testing

```bash
# Test server is responding
curl -I https://my-server.fastmcp.app/mcp

# Test tool listing
curl https://my-server.fastmcp.app/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Test specific tool
curl https://my-server.fastmcp.app/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "hello",
      "arguments": {"name": "World"}
    },
    "id": 2
  }'
```

## Common Pitfalls

### 1. Import Order Issues
```python
# ❌ Wrong: Config used before validation
from shared import Config

API_URL = Config.API_URL  # Might not be validated yet

# ✅ Right: Validate first
from shared import Config

Config.validate()  # Validate immediately
API_URL = Config.API_URL
```

### 2. Async/Sync Mixing
```python
# ❌ Wrong: Sync in async context
@mcp.tool()
async def fetch_data():
    response = requests.get(url)  # Blocking!
    return response.json()

# ✅ Right: Use async client
import httpx

@mcp.tool()
async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### 3. Global Client Issues
```python
# ❌ Wrong: Client at module level
import httpx

# This creates the client before event loop exists
client = httpx.AsyncClient()

@mcp.tool()
async def fetch_data():
    return await client.get(url)

# ✅ Right: Create client when needed
import httpx

# Store as None initially
_client = None

async def get_client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient()
    return _client

@mcp.tool()
async def fetch_data():
    client = await get_client()
    return await client.get(url)
```

### 4. Large Response Issues
```python
# ❌ Wrong: Returning huge responses
@mcp.tool()
async def get_all_data():
    # This might return MB of data
    return await fetch_entire_database()

# ✅ Right: Paginate and limit
@mcp.tool()
async def get_data(page: int = 1, limit: int = 20):
    # Return manageable chunks
    return await fetch_paginated_data(page, limit)
```

## Deployment Checklist

Before deploying to FastMCP Cloud:

- [ ] Server object (`mcp`/`server`/`app`) at module level
- [ ] All imports work without errors
- [ ] requirements.txt contains only PyPI packages
- [ ] No relative imports outside the package
- [ ] Environment variables have defaults or validation
- [ ] .env.example documents all variables
- [ ] Repository is accessible (public or FastMCP has access)
- [ ] Entry point path is correct
- [ ] Server runs locally with `python server.py`
- [ ] No global async clients created at module level
- [ ] Response sizes are reasonable (< 1MB)
- [ ] Error handling in place
- [ ] Logging configured appropriately

## Quick Fixes

### "No server object found"
```python
# Add at module level:
mcp = FastMCP(name="your-server")
# or
server = FastMCP(name="your-server")
# or
app = FastMCP(name="your-server")
```

### "Failed to install dependencies"
```bash
# Remove any -e or local references from requirements.txt
# Copy shared code directly into your project
cp -r ../shared project/src/shared
```

### "Module not found"
```python
# Fix imports - use relative imports within package
from .shared import Config  # If shared is in same directory
# or
from shared import Config  # If shared is in src/
```

### "Connection timeout"
```python
# Add timeout and retry logic
import httpx

client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0),
    limits=httpx.Limits(max_keepalive_connections=5)
)
```

## Success Metrics

Your deployment is successful when:
1. Server shows "Running" status in FastMCP Cloud
2. `/mcp` endpoint responds with 200 OK
3. Tools list returns expected tools
4. Test tool calls return valid responses
5. Logs show normal operation without errors
6. Response times are under 2 seconds
7. Memory usage stays under 80%

## Lessons Summary

1. **Always create server objects at module level**
2. **Never use local package dependencies**
3. **Embed shared code rather than reference it**
4. **Test locally in a clean environment first**
5. **Provide sensible defaults for all config**
6. **Start with minimal resources and scale up**
7. **Use monorepo for related servers**
8. **Document everything in your repository**

This guide is based on real deployment experience and will save you hours of debugging.