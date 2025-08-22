"""
Static Resources
================
Static server resources that don't change frequently.
"""

import logging
from typing import Dict, Any
from datetime import datetime

from ..utils import Config

logger = logging.getLogger(__name__)


async def server_info() -> str:
    """
    Server information resource.
    
    Returns:
        Server information as formatted text
    """
    try:
        info = f"""FastMCP Modular Server
========================

Name: {Config.SERVER_NAME}
Version: {Config.SERVER_VERSION}
Environment: {Config.ENVIRONMENT}
Transport: {Config.TRANSPORT}

Features:
- Advanced Tools: {Config.ENABLE_ADVANCED_FEATURES}
- Caching: {Config.ENABLE_CACHE}
- Rate Limiting: {getattr(Config, 'ENABLE_RATE_LIMITING', False)}

API Configuration:
- Base URL: {Config.API_BASE_URL or 'Not configured'}
- Timeout: {Config.REQUEST_TIMEOUT}s
- Max Retries: {Config.MAX_RETRIES}

Cache Configuration:
- Enabled: {Config.ENABLE_CACHE}
- TTL: {Config.CACHE_TTL}s
- Max Size: {getattr(Config, 'CACHE_MAX_SIZE', 1000)} items

Generated: {datetime.now().isoformat()}
"""
        return info
        
    except Exception as e:
        logger.error(f"Error generating server info: {e}")
        return f"Error: {str(e)}"


async def configuration() -> str:
    """
    Server configuration resource.
    
    Returns:
        Configuration as formatted JSON-like text
    """
    try:
        import json
        
        config_dict = {
            "server": {
                "name": Config.SERVER_NAME,
                "version": Config.SERVER_VERSION,
                "environment": Config.ENVIRONMENT,
                "transport": Config.TRANSPORT,
                "port": Config.PORT if Config.TRANSPORT == "http" else None
            },
            "features": {
                "advanced": Config.ENABLE_ADVANCED_FEATURES,
                "cache": Config.ENABLE_CACHE,
                "rate_limiting": getattr(Config, 'ENABLE_RATE_LIMITING', False),
                "monitoring": getattr(Config, 'ENABLE_MONITORING', False)
            },
            "api": {
                "base_url": Config.API_BASE_URL,
                "timeout": Config.REQUEST_TIMEOUT,
                "max_retries": Config.MAX_RETRIES,
                "verify_ssl": getattr(Config, 'VERIFY_SSL', True)
            },
            "cache": {
                "enabled": Config.ENABLE_CACHE,
                "ttl": Config.CACHE_TTL,
                "max_size": getattr(Config, 'CACHE_MAX_SIZE', 1000),
                "eviction_policy": getattr(Config, 'CACHE_EVICTION_POLICY', 'LRU')
            },
            "logging": {
                "level": Config.LOG_LEVEL,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": getattr(Config, 'LOG_FILE', None)
            }
        }
        
        return json.dumps(config_dict, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Error generating configuration: {e}")
        return f"Error: {str(e)}"


async def documentation() -> str:
    """
    Server documentation resource.
    
    Returns:
        Documentation as formatted markdown
    """
    try:
        docs = """# FastMCP Modular Server Documentation

## Overview
This is a modular FastMCP server implementation with organized components.

## Architecture

### Directory Structure
```
src/
├── server.py          # Main server file
├── tools/            # Tool implementations
│   ├── data_tools.py
│   ├── api_tools.py
│   ├── file_tools.py
│   ├── utility_tools.py
│   └── advanced.py
├── resources/        # Resource providers
│   ├── static.py
│   └── dynamic.py
├── prompts/          # Prompt templates
│   └── templates.py
├── shared/           # Shared utilities
│   ├── config.py
│   ├── cache.py
│   ├── api_client.py
│   └── utils.py
└── handlers/         # Client handlers
    └── events.py
```

## Available Tools

### Data Processing
- `process_data`: Process data with various operations
- `transform_data`: Transform data structures
- `validate_data`: Validate against schemas
- `export_data`: Export in multiple formats

### API Operations
- `fetch_api_data`: GET requests to APIs
- `post_api_data`: POST data to APIs
- `update_api_resource`: Update resources (PUT/PATCH)
- `delete_api_resource`: Delete resources

### File Operations
- `read_file`: Read file contents
- `write_file`: Write to files
- `list_files`: List directory contents
- `process_csv`: CSV operations

### Utilities
- `health_check`: Server health monitoring
- `get_status`: Current server status
- `run_diagnostics`: Comprehensive diagnostics

### Advanced Features
- `interactive_setup`: Configuration wizard
- `batch_processor`: Batch processing
- `ai_assistant`: AI-powered assistance

## Resources

### Static Resources
- `info://server`: Server information
- `config://settings`: Configuration details
- `docs://readme`: This documentation

### Dynamic Resources (Templates)
- `user://{user_id}/profile`: User profiles
- `project://{project_id}/data`: Project data
- `api://{version}/{endpoint}`: API endpoints

## Configuration

### Environment Variables
- `MCP_SERVER_NAME`: Server name
- `MCP_SERVER_VERSION`: Server version
- `MCP_ENVIRONMENT`: Environment (dev/staging/prod)
- `MCP_LOG_LEVEL`: Logging level
- `MCP_API_BASE_URL`: API base URL
- `MCP_API_KEY`: API authentication key
- `MCP_CACHE_ENABLED`: Enable caching
- `MCP_CACHE_TTL`: Cache TTL in seconds

### Configuration File
Create a `config.json` file:

```json
{
  "server": {
    "name": "my-server",
    "version": "1.0.0"
  },
  "api": {
    "base_url": "https://api.example.com",
    "api_key": "your-api-key"
  },
  "cache": {
    "enabled": true,
    "ttl": 300
  }
}
```

## Usage

### Starting the Server
```bash
# Standard mode
python src/server.py

# Debug mode
python src/server.py --debug

# Test mode
python src/server.py --test
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
CMD ["python", "src/server.py"]
```

## API Integration

### Authentication
The server supports multiple authentication methods:
- API Key (header or query parameter)
- Bearer Token
- OAuth2 (if configured)

### Rate Limiting
Default limits:
- 100 requests per minute per client
- 1000 requests per hour per client

### Error Handling
All errors follow the standard format:
```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Monitoring

### Health Check Endpoint
`GET /health` returns:
- Server status
- Resource usage
- Dependency checks
- Cache statistics

### Metrics
Available metrics:
- Request count
- Response time
- Error rate
- Cache hit ratio

## Extending the Server

### Adding New Tools
1. Create a new file in `src/tools/`
2. Implement async functions
3. Import in `src/tools/__init__.py`
4. Register in `src/server.py`

### Adding Resources
1. Add function to `src/resources/static.py` or `dynamic.py`
2. Import in `src/resources/__init__.py`
3. Register with appropriate URI pattern

### Custom Handlers
1. Create handler in `src/handlers/`
2. Implement event callbacks
3. Register with server lifecycle

## Support

For issues or questions:
- GitHub: https://github.com/yourusername/fastmcp-modular
- Documentation: https://docs.example.com
- Email: support@example.com

---
Generated: {datetime.now().isoformat()}
"""
        return docs
        
    except Exception as e:
        logger.error(f"Error generating documentation: {e}")
        return f"Error: {str(e)}"