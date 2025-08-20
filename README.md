# FastMCP Documentation Repository

> Comprehensive documentation and reference guides for developing FastMCP (Model Context Protocol) servers

## 📚 Documentation Overview

This repository contains extensive documentation for FastMCP v2, including all advanced features, patterns, integrations, and troubleshooting guides.

### Core Documentation

- **[FASTMCP_GUIDE.md](FASTMCP_GUIDE.md)** - Complete development guide (700+ lines)
- **[FASTMCP_REFERENCE.md](FASTMCP_REFERENCE.md)** - Quick reference with CLI commands and patterns
- **[FASTMCP_ADVANCED_FEATURES.md](FASTMCP_ADVANCED_FEATURES.md)** - Comprehensive guide to v2 features
- **[FASTMCP_PATTERNS.md](FASTMCP_PATTERNS.md)** - Common patterns and solutions
- **[FASTMCP_INTEGRATIONS.md](FASTMCP_INTEGRATIONS.md)** - API integration guides (Gemini, OpenAI, etc.)
- **[FASTMCP_TROUBLESHOOTING.md](FASTMCP_TROUBLESHOOTING.md)** - Debugging and problem-solving guide

### Strategy & Learning

- **[DEPLOYMENT_LEARNINGS.md](DEPLOYMENT_LEARNINGS.md)** - Deployment insights and cloud strategies
- **[MONOREPO_STRATEGY.md](MONOREPO_STRATEGY.md)** - Multi-server repository patterns

### Project Templates

We provide two template structures for different needs:

#### 1. Simple Template (`project_template/`)
Best for small to medium servers with straightforward functionality:

- **server.py** - Basic MCP server template
- **server_advanced.py** - Advanced server with all v2 features
- **handlers.py** - Client handler implementations
- **requirements.txt** - Dependency management
- **.env.example** - Environment configuration template

#### 2. Modular Template (`modular_template/`)
Production-ready structure for complex servers with separation of concerns:

- **src/server.py** - Main entry point with lifecycle hooks
- **src/tools/** - Organized tool modules (data, API, file, utility)
- **src/resources/** - Static and dynamic resources with templates
- **src/prompts/** - Pre-defined prompt templates
- **src/handlers/** - Client and event handlers
- **src/shared/** - Utilities (config, cache, API client)
- **pyproject.toml** - Modern Python project configuration
- **.env.example** - Comprehensive environment variables

Choose the modular template when you need:
- Multiple tool categories with many functions
- Complex resource management with caching
- Client connection handling
- Event-driven architecture
- Shared utilities across modules

## 🚀 FastMCP v2 Features Covered

### Advanced Features
- ✅ Resource Templates (dynamic resources with parameters)
- ✅ Elicitation (interactive user input during execution)
- ✅ Progress Tracking (monitoring long-running operations)
- ✅ Sampling (LLM integration for servers)
- ✅ Client Handlers (elicitation, progress, sampling)
- ✅ Tool Transformation (enhance existing tools)
- ✅ Server Composition (mount sub-servers)

### Integrations
- ✅ OpenAPI/Swagger (auto-generate from specs)
- ✅ FastAPI (convert FastAPI apps to MCP)
- ✅ Gemini SDK (Google AI integration)
- ✅ OpenAI API (GPT integration)
- ✅ Anthropic/Claude (Claude integration)
- ✅ Authentication (OAuth2, Bearer, JWT)
- ✅ Permit.io (authorization)

## 📖 Quick Start

### For New Projects

1. Start with [FASTMCP_GUIDE.md](FASTMCP_GUIDE.md) for fundamentals
2. Use [project_template/](project_template/) as your starting point
3. Reference [FASTMCP_PATTERNS.md](FASTMCP_PATTERNS.md) for common solutions

### For Advanced Features

1. Review [FASTMCP_ADVANCED_FEATURES.md](FASTMCP_ADVANCED_FEATURES.md)
2. Check [project_template/server_advanced.py](project_template/server_advanced.py) for examples
3. Implement handlers from [project_template/handlers.py](project_template/handlers.py)

### For API Integration

1. Read [FASTMCP_INTEGRATIONS.md](FASTMCP_INTEGRATIONS.md)
2. Choose your integration approach (OpenAPI, direct, or FastAPI)
3. Follow authentication patterns for secure connections

### For Troubleshooting

1. Check [FASTMCP_TROUBLESHOOTING.md](FASTMCP_TROUBLESHOOTING.md)
2. Enable debug logging
3. Use the test harness examples

## 🎯 Use Cases

This documentation serves as:

- **Reference Guide** - Quick lookup for syntax and patterns
- **Learning Resource** - Understand FastMCP capabilities
- **Project Template** - Starting point for new servers
- **Troubleshooting Guide** - Debug common issues
- **Integration Manual** - Connect to external APIs
- **Best Practices** - Production-ready patterns

## 🛠️ Example: Creating an Advanced Server

```python
from fastmcp import FastMCP, elicit, report_progress, sample

mcp = FastMCP("Advanced Server")

# Resource template
@mcp.resource("user://{user_id}/profile")
async def get_user(user_id: str):
    return {"id": user_id, "name": f"User {user_id}"}

# Tool with elicitation
@mcp.tool
async def interactive_setup():
    name = await elicit("Project name?", str)
    return {"project": name}

# Tool with progress
@mcp.tool
async def process_data(items: list):
    for i, item in enumerate(items):
        await report_progress(i, len(items), f"Processing {item}")
    return {"processed": len(items)}

# Tool with LLM sampling
@mcp.tool
async def ai_generate(prompt: str):
    result = await sample(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4"
    )
    return {"generated": result["content"]}

if __name__ == "__main__":
    mcp.run()
```

## 📚 Documentation Structure

```
fastmcp-documentation/
├── README.md                        # This file
├── FASTMCP_GUIDE.md                # Complete development guide
├── FASTMCP_REFERENCE.md            # Quick reference
├── FASTMCP_ADVANCED_FEATURES.md    # V2 features deep dive
├── FASTMCP_PATTERNS.md             # Common patterns
├── FASTMCP_INTEGRATIONS.md         # API integration guides
├── FASTMCP_TROUBLESHOOTING.md      # Debugging guide
├── DEPLOYMENT_LEARNINGS.md         # Deployment insights
├── MONOREPO_STRATEGY.md           # Multi-server patterns
├── project_template/               # Simple template for basic servers
│   ├── server.py                   # Basic server
│   ├── server_advanced.py          # Advanced features demo
│   ├── handlers.py                 # Client handlers
│   ├── requirements.txt            # Dependencies
│   └── .env.example               # Environment template
└── modular_template/              # Modular template for complex servers
    ├── src/
    │   ├── server.py              # Main entry point
    │   ├── tools/                 # Tool modules
    │   ├── resources/             # Resource modules
    │   ├── prompts/               # Prompt templates
    │   ├── handlers/              # Event handlers
    │   └── shared/                # Shared utilities
    ├── pyproject.toml             # Project configuration
    └── .env.example               # Environment template
```

## 🔗 Resources

- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [FastMCP Documentation](https://docs.fastmcp.com)
- [FastMCP Cloud](https://fastmcp.cloud)
- [MCP Protocol](https://modelcontextprotocol.io)

## 📝 License

This documentation is for internal reference and development guidance.

## 🤝 Contributing

This is a private documentation repository. Updates should focus on:
- Documenting new FastMCP features
- Adding integration examples
- Improving troubleshooting guides
- Sharing deployment learnings

---

*Last Updated: August 2025*
*FastMCP Version: 0.3.0+*