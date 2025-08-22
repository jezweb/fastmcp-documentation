# FastMCP Server Template

A production-ready template for building FastMCP servers.

## Features

- ✅ Complete project structure
- ✅ Self-contained single-file design
- ✅ Environment variable management
- ✅ API integration support
- ✅ Error handling and logging
- ✅ Type hints and documentation
- ✅ Ready for deployment

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update with your values:

```bash
cp .env.example .env
```

### 3. Run the Server

```bash
# Run with default STDIO transport
python server.py

# Run with HTTP transport
python server.py --transport http --port 8000

# Run in test mode
python server.py --test
```

## Project Structure

```
.
├── server.py           # Main server implementation
├── requirements.txt    # Python dependencies
├── .env.example       # Environment variable template
├── .env               # Your local environment (git-ignored)
├── .gitignore         # Git ignore patterns
└── README.md          # This file
```

## Deployment

### FastMCP Cloud

1. Push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
gh repo create my-mcp-server --public
git push -u origin main
```

2. Deploy on [fastmcp.cloud](https://fastmcp.cloud):
   - Sign in with GitHub
   - Create new project
   - Select your repository
   - Configure and deploy

### Local Development

Use the FastMCP CLI for development:

```bash
# Install in Claude Desktop
fastmcp install claude-desktop server.py

# Development mode with inspector
fastmcp dev server.py

# Test with client
python -c "
import asyncio
from fastmcp import Client

async def test():
    async with Client('server.py') as client:
        tools = await client.list_tools()
        print(f'Tools: {[t.name for t in tools]}')

asyncio.run(test())
"
```

## Available Components

### Tools
- `example_tool` - Demonstrates tool creation with optional parameters
- `process_data` - Shows async data processing with different operations

### Resources
- `info://status` - Server status and health information
- `data://config` - Server configuration (non-sensitive)
- `data://items/{item_id}` - Dynamic resource template example

### Prompts
- `help` - Generate help text for the server
- `analyze` - Generate analysis prompts for topics

## Configuration

Environment variables are loaded from `.env`:

- `SERVER_NAME` - Name of your MCP server
- `API_BASE_URL` - Base URL for API integration
- `API_KEY` - API authentication key
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

See `.env.example` for all available options.

## API Integration

To integrate with an external API:

1. Add API credentials to `.env`:
```env
API_BASE_URL=https://api.example.com
API_KEY=your-api-key
```

2. The server includes an optional `api_request` tool when configured

3. For OpenAPI/Swagger integration, see the examples in the parent directory

## Testing

Run the built-in test mode:

```bash
python server.py --test
```

This will:
- List all available tools
- Test the example_tool
- Display available resources

## License

MIT

## Support

- [FastMCP Documentation](https://docs.fastmcp.com)
- [GitHub Issues](https://github.com/jlowin/fastmcp/issues)
- [FastMCP Cloud](https://fastmcp.cloud)