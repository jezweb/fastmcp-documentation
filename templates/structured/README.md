# FastMCP Structured Server Template

A production-ready structured template for FastMCP servers with organized modules and self-contained utilities.

## Structure

```
structured_template/
├── src/
│   ├── server.py           # Main server entry point
│   ├── utils.py            # Self-contained utilities
│   ├── tools/              # Tool implementations
│   │   ├── __init__.py
│   │   ├── data_tools.py   # Data processing tools
│   │   ├── api_tools.py    # API integration tools
│   │   ├── file_tools.py   # File operation tools
│   │   ├── utility_tools.py # Utility tools
│   │   └── advanced.py     # Advanced feature tools
│   ├── resources/          # Resource definitions
│   │   ├── __init__.py
│   │   ├── static.py       # Static resources
│   │   └── dynamic.py      # Dynamic resources with templates
│   ├── prompts/            # Prompt templates
│   │   ├── __init__.py
│   │   └── templates.py    # Pre-defined prompt templates
│   └── handlers/           # Event and client handlers
│       ├── __init__.py
│       ├── client_handlers.py  # Client event handling
│       └── event_handlers.py   # Server lifecycle events
├── pyproject.toml          # Project configuration
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Features

### Structured Architecture
- **Separation of Concerns**: Tools, resources, prompts, and handlers in separate modules
- **Self-Contained Design**: All utilities consolidated in `utils.py` for easy deployment
- **Clean Imports**: Organized `__init__.py` files with explicit exports

### Advanced Features
- **Lifecycle Hooks**: Server startup/shutdown handlers
- **Client Handlers**: Manage client connections and messages
- **Event System**: Request/response middleware and event handling
- **Caching**: LRU cache with TTL support
- **API Client**: Connection pooling and retry logic
- **Configuration**: Environment-based configuration with validation
- **Self-Contained Utils**: All utilities in single `utils.py` file for simple deployment

### Tools Module
- Data processing tools (transform, validate, export)
- API integration tools (CRUD operations)
- File operation tools (read, write, process)
- Utility tools (system info, calculations)
- Advanced features (sampling, progress tracking)

### Resources Module
- Static resources (configuration, documentation)
- Dynamic resources with URI templates
- Cached resource responses
- Analytics and reporting endpoints

### Prompts Module
- Pre-built prompt templates for common tasks
- Analysis, summary, debugging prompts
- Code review and documentation templates
- Context-aware prompt generation

### Handlers Module
- Client connection management
- Error handling and recovery
- Server lifecycle events
- Request/response middleware

## Usage

1. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install Dependencies**
   ```bash
   pip install -e .
   ```

3. **Run Server**
   ```bash
   python src/server.py
   ```

## Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

Key configuration areas:
- **Server Settings**: Name, version, environment
- **Transport**: stdio or HTTP mode
- **API Integration**: Base URL, authentication, timeouts
- **Cache**: TTL, size limits, eviction policy
- **Security**: Authentication, CORS, rate limiting

## Development

### Adding New Tools

1. Create a new function in the appropriate tools file
2. Use async/await pattern
3. Return standardized responses using `format_success`/`format_error`
4. Add to module exports in `__init__.py`

Example:
```python
from ..utils import format_success, format_error, logger

async def my_new_tool(param1: str, param2: int) -> Dict[str, Any]:
    try:
        # Tool implementation
        result = await process_data(param1, param2)
        return format_success(result, "Operation completed")
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return format_error(e, "TOOL_ERROR")
```

### Adding New Resources

1. Define resource function in `resources/static.py` or `dynamic.py`
2. Use caching for expensive operations
3. Support URI templates for dynamic resources
4. Register in server.py

Example:
```python
from ..utils import cache_get, cache_set
import json

async def get_user_data(user_id: str) -> str:
    cache_key = f"user_{user_id}"
    cached = await cache_get(cache_key)
    if cached:
        return cached
    
    # Fetch data
    data = await fetch_user_data(user_id)
    await cache_set(cache_key, data, ttl=300)
    return json.dumps(data)
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src

# Format code
black src/
ruff src/
```

## Best Practices

1. **Error Handling**: Always use try/except blocks and return formatted errors
2. **Logging**: Use appropriate log levels (debug, info, warning, error)
3. **Caching**: Cache expensive operations with appropriate TTLs
4. **Validation**: Validate inputs using the validation utilities
5. **Documentation**: Keep docstrings updated with parameter descriptions
6. **Type Hints**: Use type hints for all function parameters and returns

## Deployment

1. Set production environment variables
2. Configure appropriate log levels
3. Enable monitoring and rate limiting if needed
4. Use connection pooling for API integrations
5. Set up health checks and monitoring

## Self-Contained Design

This template uses a self-contained architecture where all utilities are consolidated in `utils.py`. This approach:
- Eliminates complex import path issues
- Makes deployment simple and reliable
- Works seamlessly with FastMCP Cloud
- Follows the pattern used in production servers like SimPro MCP

All tool modules import from `utils.py` using relative imports like `from ..utils import Config, format_success`.

## License

MIT