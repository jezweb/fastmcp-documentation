# FastMCP Documentation

> Streamlined documentation for building and deploying FastMCP servers to [FastMCP Cloud](https://fastmcp.cloud)

## 🚀 Quick Start

### 1. Install FastMCP
```bash
pip install fastmcp
```

### 2. Create Your Server
```python
from fastmcp import FastMCP

# MUST be at module level for FastMCP Cloud
mcp = FastMCP("my-server")

@mcp.tool()
async def hello(name: str) -> str:
    """Say hello to someone."""
    return f"Hello, {name}!"
```

### 3. Deploy to FastMCP Cloud
1. Push your code to GitHub
2. Connect repository at [fastmcp.cloud](https://fastmcp.cloud)
3. Add environment variables
4. Deploy with one click

## 📚 Documentation

### Essential Guides
- **[docs/GUIDE.md](docs/GUIDE.md)** - Complete development guide
- **[docs/REFERENCE.md](docs/REFERENCE.md)** - Quick CLI reference
- **[docs/CLOUD_DEPLOYMENT.md](docs/CLOUD_DEPLOYMENT.md)** - FastMCP Cloud deployment guide

### Feature Documentation
- **[docs/FEATURES.md](docs/FEATURES.md)** - Advanced FastMCP v2 features
- **[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)** - API integration patterns
- **[docs/PATTERNS.md](docs/PATTERNS.md)** - Common patterns and solutions

### Development Tools
- **[docs/CLI.md](docs/CLI.md)** - CLI command reference
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Debugging guide

## 🎯 Templates

### Simple Template
For straightforward servers with basic functionality:
```
templates/simple/
├── server.py              # Basic server implementation
├── server_advanced.py     # Advanced features example
├── handlers.py           # Client handlers
└── requirements.txt      # Dependencies
```

### Modular Template
Production-ready structure matching our SimPro pattern:
```
templates/modular/
├── src/
│   ├── server.py         # Main entry point
│   ├── tools/           # Organized tool modules
│   ├── resources/       # Resource definitions
│   ├── prompts/         # Prompt templates
│   └── shared/          # Shared utilities
└── requirements.txt
```

## ⚡ FastMCP Cloud Requirements

### Critical for Cloud Deployment
1. **Module-level server object** named `mcp`, `server`, or `app`
2. **PyPI dependencies only** in requirements.txt
3. **Public GitHub repository** (or accessible to FastMCP Cloud)
4. **Environment variables** for configuration

### Example Cloud-Ready Server
```python
# server.py
from fastmcp import FastMCP
import os

# MUST be at module level
mcp = FastMCP(
    name="production-server"
)

# Use environment variables
API_KEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

@mcp.tool()
async def production_tool(data: str) -> dict:
    """Production-ready tool."""
    # Your implementation here
    return {"status": "success", "data": data}

# Optional: for local testing
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run())
```

## 🔧 Common Use Cases

### API Integration Server
```python
from fastmcp import FastMCP
import httpx

mcp = FastMCP("api-server")

@mcp.tool()
async def fetch_data(endpoint: str) -> dict:
    """Fetch data from API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/{endpoint}")
        return response.json()
```

### Data Processing Server
```python
from fastmcp import FastMCP

mcp = FastMCP("data-processor")

@mcp.tool()
async def process_csv(data: str) -> dict:
    """Process CSV data."""
    rows = data.split('\n')
    return {"rows": len(rows), "processed": True}
```

### AI Integration Server
```python
from fastmcp import FastMCP
from openai import AsyncOpenAI

mcp = FastMCP("ai-server")
client = AsyncOpenAI()

@mcp.tool()
async def generate_text(prompt: str) -> str:
    """Generate text using AI."""
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## 🏗️ Project Structure

```
fastmcp-documentation/
├── README.md              # This file
├── docs/                  # Core documentation
│   ├── GUIDE.md          # Development guide
│   ├── REFERENCE.md      # Quick reference
│   ├── CLOUD_DEPLOYMENT.md # Cloud deployment
│   ├── FEATURES.md       # Advanced features
│   ├── INTEGRATIONS.md   # API integrations
│   ├── PATTERNS.md       # Common patterns
│   ├── CLI.md           # CLI reference
│   └── TROUBLESHOOTING.md # Debug guide
├── templates/            # Project templates
│   ├── simple/          # Basic server template
│   └── modular/         # Production template
└── complex-detail/      # Advanced topics (self-hosting, infrastructure)
```

## 🌟 FastMCP v2 Features

- ✅ **Resource Templates** - Dynamic resources with parameters
- ✅ **Elicitation** - Interactive user input during execution
- ✅ **Progress Tracking** - Monitor long-running operations
- ✅ **Sampling** - LLM integration for servers
- ✅ **Client Handlers** - Handle client events
- ✅ **Tool Transformation** - Enhance existing tools
- ✅ **Server Composition** - Mount sub-servers

## 🚦 Getting Started Steps

### 1. Choose a Template
- **Simple**: For basic servers with few tools
- **Modular**: For production servers with multiple features

### 2. Develop Locally
```bash
# Install FastMCP
pip install fastmcp

# Test your server
fastmcp dev server.py
```

### 3. Deploy to Cloud
- Push to GitHub
- Connect at [fastmcp.cloud](https://fastmcp.cloud)
- Configure environment variables
- Deploy instantly

## 🔍 Troubleshooting

### Common Issues

**"No server object found"**
- Ensure your server object is at module level
- Name it `mcp`, `server`, or `app`

**"Module not found"**
- Use only PyPI packages in requirements.txt
- No local packages or git URLs

**"Environment variable not set"**
- Add variables in FastMCP Cloud dashboard
- Use `os.getenv()` with defaults

## 🛠️ Advanced Topics

For advanced deployment scenarios, see the `complex-detail/` folder:
- Self-hosting with Docker/Kubernetes
- Cloud infrastructure (AWS, Azure, GCP)
- CI/CD pipelines
- Monitoring and scaling

## 🔗 Resources

- [FastMCP Cloud](https://fastmcp.cloud) - Deploy your servers
- [FastMCP GitHub](https://github.com/jlowin/fastmcp) - Source code
- [MCP Protocol](https://modelcontextprotocol.io) - Protocol specification
- Context 7 official docs library: context7CompatibleLibraryID: "/jlowin/fastmcp"

## 📝 License

This documentation is for internal reference and development guidance.

---

*Last Updated: August 2025*
*FastMCP Version: 2.12.0+*
