# FastMCP Monorepo Strategy for Multi-Server Projects

This guide covers strategies for organizing and deploying multiple related MCP servers from a single repository, based on the successful SimPro MCP servers deployment.

## Table of Contents
1. [When to Use Monorepo](#when-to-use-monorepo)
2. [Repository Structure](#repository-structure)
3. [Shared Code Management](#shared-code-management)
4. [Dependency Management](#dependency-management)
5. [Deployment Strategies](#deployment-strategies)
6. [Development Workflow](#development-workflow)
7. [Testing Strategy](#testing-strategy)
8. [CI/CD Configuration](#cicd-configuration)
9. [Cost Optimization](#cost-optimization)
10. [Real-World Example](#real-world-example)

## When to Use Monorepo

### Ideal Use Cases
- **Related servers** that share domain logic (e.g., all connect to same API)
- **Modular functionality** that can be deployed separately
- **Client-specific deployments** where different clients need different server combinations
- **Shared utilities** and configuration across servers
- **Consistent versioning** needed across all servers

### When NOT to Use Monorepo
- Servers with completely different dependencies
- Different deployment schedules or release cycles
- Different teams maintaining different servers
- Security requirements demanding isolation

## Repository Structure

### Recommended Layout
```
my-mcp-servers/
├── README.md                    # Project overview
├── DEPLOYMENT.md               # Deployment instructions
├── TROUBLESHOOTING.md         # Common issues
├── .gitignore                  # Git ignore patterns
├── .github/                    # GitHub specific files
│   └── workflows/              # CI/CD workflows
│       ├── test.yml           # Test all servers
│       └── deploy.yml         # Deployment automation
│
├── shared/                     # Shared utilities (for development)
│   ├── __init__.py
│   ├── config.py              # Shared configuration
│   ├── client.py              # Shared API client
│   ├── utils.py               # Common utilities
│   └── models.py              # Shared data models
│
├── scripts/                    # Maintenance scripts
│   ├── copy-shared.sh         # Copy shared to all servers
│   ├── test-all.sh           # Test all servers
│   └── deploy-all.sh         # Deploy all servers
│
├── server-one/
│   ├── src/
│   │   ├── server.py          # Server entry point
│   │   └── shared/            # Copied shared utilities
│   │       ├── __init__.py
│   │       ├── config.py
│   │       └── utils.py
│   ├── requirements.txt       # Server-specific deps
│   ├── tests/                 # Server tests
│   └── README.md              # Server documentation
│
├── server-two/
│   ├── src/
│   │   ├── server.py
│   │   └── shared/            # Same shared utilities
│   ├── requirements.txt
│   ├── tests/
│   └── README.md
│
└── server-three/
    ├── src/
    │   ├── server.py
    │   └── shared/
    ├── requirements.txt
    ├── tests/
    └── README.md
```

## Shared Code Management

### Strategy 1: Copy During Development (Recommended for FastMCP Cloud)
```bash
#!/bin/bash
# scripts/copy-shared.sh

# Copy shared utilities to all servers
for server in server-*/; do
    echo "Copying shared to $server"
    rm -rf "$server/src/shared"
    cp -r shared "$server/src/shared"
done
```

**Pros:**
- Works with FastMCP Cloud (no local dependencies)
- Each server is self-contained
- Simple deployment

**Cons:**
- Code duplication
- Need to remember to copy after changes

### Strategy 2: Symbolic Links (Local Development Only)
```bash
#!/bin/bash
# scripts/link-shared.sh

for server in server-*/; do
    ln -sf "../../shared" "$server/src/shared"
done
```

**Pros:**
- No code duplication
- Changes immediately visible

**Cons:**
- Doesn't work with FastMCP Cloud
- Can cause issues with some tools

### Strategy 3: Build-Time Injection
```python
# build.py
import shutil
import os

def build_server(server_name):
    """Build server with embedded shared code."""
    # Copy server code
    shutil.copytree(f"{server_name}/src", f"build/{server_name}")
    
    # Inject shared code
    shutil.copytree("shared", f"build/{server_name}/shared")
    
    # Update imports
    update_imports(f"build/{server_name}")
```

## Dependency Management

### Shared Dependencies Pattern
```python
# shared/requirements-base.txt
fastmcp>=2.12.0
httpx>=0.24.0
python-dotenv>=1.0.0
pydantic>=2.0.0
```

```python
# server-one/requirements.txt
# Include base requirements
-r ../shared/requirements-base.txt

# Server-specific dependencies
pandas>=2.0.0
```

### For FastMCP Cloud Deployment
```python
# server-one/requirements.txt
# Must list all dependencies explicitly
fastmcp>=2.12.0
httpx>=0.24.0
python-dotenv>=1.0.0
pydantic>=2.0.0
pandas>=2.0.0  # Server-specific
```

## Deployment Strategies

### Strategy 1: Deploy All Servers
```yaml
# FastMCP Cloud configuration for each server
servers:
  - name: myapp-server-one
    repository: username/my-mcp-servers
    entrypoint: server-one/src/server.py
    requirements: server-one/requirements.txt
    
  - name: myapp-server-two
    repository: username/my-mcp-servers
    entrypoint: server-two/src/server.py
    requirements: server-two/requirements.txt
```

### Strategy 2: Selective Deployment by Client Tier
```python
# deployment-config.json
{
  "tiers": {
    "basic": ["server-one"],
    "professional": ["server-one", "server-two"],
    "enterprise": ["server-one", "server-two", "server-three"]
  },
  "clients": {
    "client-a": "basic",
    "client-b": "professional",
    "client-c": "enterprise"
  }
}
```

### Strategy 3: Feature Flags
```python
# shared/config.py
import os

class Features:
    ENABLE_ADVANCED = os.getenv("ENABLE_ADVANCED", "false").lower() == "true"
    ENABLE_REPORTING = os.getenv("ENABLE_REPORTING", "false").lower() == "true"
    ENABLE_AUTOMATION = os.getenv("ENABLE_AUTOMATION", "false").lower() == "true"

# server.py
from shared.config import Features

if Features.ENABLE_ADVANCED:
    @mcp.tool()
    async def advanced_feature():
        pass
```

## Development Workflow

### Local Development Setup
```bash
# 1. Clone repository
git clone https://github.com/username/my-mcp-servers.git
cd my-mcp-servers

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Copy shared utilities
./scripts/copy-shared.sh

# 5. Run specific server
cd server-one
python src/server.py
```

### Development Scripts
```bash
#!/bin/bash
# scripts/dev.sh - Start server in development mode

SERVER=$1
if [ -z "$SERVER" ]; then
    echo "Usage: ./scripts/dev.sh <server-name>"
    exit 1
fi

# Copy latest shared code
cp -r shared "$SERVER/src/shared"

# Run with FastMCP dev mode
cd "$SERVER"
fastmcp dev src/server.py
```

### Testing All Servers
```bash
#!/bin/bash
# scripts/test-all.sh

# Copy shared to all servers
./scripts/copy-shared.sh

# Test each server
for server in server-*/; do
    echo "Testing $server"
    cd "$server"
    python -m pytest tests/ || exit 1
    cd ..
done

echo "All tests passed!"
```

## Testing Strategy

### Shared Test Utilities
```python
# tests/shared/fixtures.py
import pytest
from fastmcp.testing import create_test_client

@pytest.fixture
def mock_api_response():
    """Mock API response for testing."""
    return {
        "status": "success",
        "data": {"test": "data"}
    }

@pytest.fixture
async def test_client(server):
    """Create test client for server."""
    async with create_test_client(server) as client:
        yield client
```

### Server-Specific Tests
```python
# server-one/tests/test_tools.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.server import mcp
from tests.shared.fixtures import test_client

@pytest.mark.asyncio
async def test_server_one_tool(test_client):
    """Test server-one specific tool."""
    result = await test_client.call_tool(
        "specific_tool",
        {"param": "test"}
    )
    assert result.success
```

### Integration Tests
```python
# tests/integration/test_cross_server.py
import asyncio
from fastmcp.testing import create_test_client

@pytest.mark.asyncio
async def test_server_interaction():
    """Test interaction between servers."""
    async with create_test_client("server-one/src/server.py") as client1:
        async with create_test_client("server-two/src/server.py") as client2:
            # Test cross-server functionality
            data = await client1.call_tool("export_data", {})
            result = await client2.call_tool("import_data", {"data": data})
            assert result.success
```

## CI/CD Configuration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test All Servers

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        server: [server-one, server-two, server-three]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r ${{ matrix.server }}/requirements.txt
        pip install pytest pytest-asyncio
    
    - name: Copy shared utilities
      run: |
        cp -r shared ${{ matrix.server }}/src/shared
    
    - name: Run tests
      run: |
        cd ${{ matrix.server }}
        python -m pytest tests/
```

### Deployment Automation
```yaml
# .github/workflows/deploy.yml
name: Deploy to FastMCP Cloud

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Copy shared to all servers
      run: ./scripts/copy-shared.sh
    
    - name: Notify FastMCP Cloud
      run: |
        # FastMCP Cloud auto-deploys on push
        echo "Deployment triggered by push to main"
```

## Cost Optimization

### Tiered Deployment Strategy
```python
# deployment/tier_manager.py
class TierManager:
    TIERS = {
        "starter": {
            "servers": ["core"],
            "resources": {"cpu": 1, "memory": "2GB"},
            "cost": "$10/month"
        },
        "professional": {
            "servers": ["core", "advanced", "reporting"],
            "resources": {"cpu": 2, "memory": "4GB"},
            "cost": "$50/month"
        },
        "enterprise": {
            "servers": ["core", "advanced", "reporting", "automation", "admin"],
            "resources": {"cpu": 4, "memory": "8GB"},
            "cost": "$200/month"
        }
    }
    
    @classmethod
    def get_client_servers(cls, client_tier):
        return cls.TIERS[client_tier]["servers"]
```

### Resource Sharing
```python
# shared/resource_pool.py
class ResourcePool:
    """Share expensive resources across servers."""
    
    _connections = {}
    
    @classmethod
    async def get_connection(cls, key: str):
        if key not in cls._connections:
            cls._connections[key] = await create_connection(key)
        return cls._connections[key]
```

### Dynamic Scaling
```python
# shared/autoscale.py
import psutil

class AutoScaler:
    @staticmethod
    async def check_load():
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        
        if cpu > 80 or memory > 80:
            # Trigger scale up
            await notify_scaling_needed()
        elif cpu < 20 and memory < 20:
            # Consider scale down
            await notify_scaling_possible()
```

## Real-World Example

### SimPro MCP Servers Structure
```
simpro-mcp-servers/
├── README.md                   # Project overview with 140 tools
├── DEPLOYMENT.md              # FastMCP Cloud deployment guide
├── TROUBLESHOOTING.md        # Common issues and solutions
│
├── simpro-shared/             # Shared utilities (dev only)
│   ├── src/
│   │   ├── config.py         # SimPro API configuration
│   │   ├── client.py         # SimPro API client
│   │   └── utils.py          # Shared utilities
│   └── setup.py
│
├── simpro-field-ops/          # Field operations (18 tools)
│   ├── src/
│   │   ├── server.py         # Module-level mcp object
│   │   └── shared/           # Embedded shared code
│   └── requirements.txt      # PyPI dependencies only
│
├── simpro-customer/           # Customer management (25 tools)
│   ├── src/
│   │   ├── server.py
│   │   └── shared/
│   └── requirements.txt
│
├── simpro-finance/            # Financial operations (22 tools)
├── simpro-inventory/          # Inventory management (20 tools)
├── simpro-workforce/          # HR and workforce (25 tools)
└── simpro-admin/              # System administration (30 tools)
```

### Deployment Configuration
```python
# Each server deployed separately to FastMCP Cloud
deployments = [
    {
        "name": "simpro-field-ops",
        "url": "https://simpro-field-ops.fastmcp.app/mcp",
        "entrypoint": "simpro-field-ops/src/server.py",
        "requirements": "simpro-field-ops/requirements.txt"
    },
    # ... repeat for each server
]
```

### Key Learnings Applied
1. **Module-level server objects** - Required for FastMCP Cloud
2. **Embedded shared code** - No local package references
3. **Self-contained servers** - Each can deploy independently
4. **Consistent structure** - Same pattern across all servers
5. **Clear documentation** - Deployment guide for each server

## Best Practices Summary

### Do's
✅ Keep servers focused on specific functionality
✅ Share code by copying, not referencing
✅ Use consistent structure across all servers
✅ Document deployment for each server
✅ Test servers both individually and together
✅ Use environment variables for configuration
✅ Version all servers together

### Don'ts
❌ Don't use local package references (-e)
❌ Don't wrap server creation in functions
❌ Don't share mutable state between servers
❌ Don't assume servers will be deployed together
❌ Don't hardcode configuration values
❌ Don't mix different deployment strategies

## Migration Guide

### Converting Single Server to Monorepo
```bash
# 1. Create new structure
mkdir my-mcp-servers
cd my-mcp-servers

# 2. Move existing server
mv ../old-server server-one

# 3. Extract shared code
mkdir shared
mv server-one/src/common/* shared/

# 4. Create copy script
cat > scripts/copy-shared.sh << 'EOF'
#!/bin/bash
for server in server-*/; do
    cp -r shared "$server/src/shared"
done
EOF

# 5. Update imports
# Change: from common import utils
# To: from shared import utils
```

### Adding New Server to Monorepo
```bash
# 1. Create server structure
mkdir server-new
mkdir -p server-new/src
mkdir server-new/tests

# 2. Copy shared utilities
cp -r shared server-new/src/shared

# 3. Create server.py
cat > server-new/src/server.py << 'EOF'
from fastmcp import FastMCP
from shared import config, utils

mcp = FastMCP(name="server-new", version="1.0.0")

@mcp.tool()
async def new_tool():
    pass

if __name__ == "__main__":
    mcp.run()
EOF

# 4. Create requirements.txt
cat > server-new/requirements.txt << 'EOF'
fastmcp>=2.12.0
httpx>=0.24.0
python-dotenv>=1.0.0
EOF

# 5. Deploy to FastMCP Cloud
# Configure with entrypoint: server-new/src/server.py
```

## Conclusion

The monorepo strategy works exceptionally well for related MCP servers that:
- Share domain logic or API connections
- Need consistent versioning
- Benefit from shared utilities
- Deploy to the same platform (FastMCP Cloud)

The key to success is maintaining self-contained servers while sharing code through copying rather than references, ensuring compatibility with FastMCP Cloud's deployment requirements.