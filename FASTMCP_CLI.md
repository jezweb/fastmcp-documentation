# FastMCP CLI Reference Documentation

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Basic Commands](#basic-commands)
- [Server Management](#server-management)
- [Development Commands](#development-commands)
- [Configuration](#configuration)
- [Testing Commands](#testing-commands)
- [Deployment Commands](#deployment-commands)
- [Utility Commands](#utility-commands)
- [Environment Variables](#environment-variables)
- [Configuration Files](#configuration-files)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Overview

The FastMCP CLI provides a comprehensive set of commands for developing, testing, and deploying MCP servers. It includes development servers, production runners, testing utilities, and management tools.

## Installation

### Install FastMCP

```bash
# Using pip
pip install fastmcp

# Using uv (recommended)
uv pip install fastmcp

# Development installation
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check version
fastmcp --version
# Output: fastmcp version 0.2.0

# Get help
fastmcp --help

# Check installation
fastmcp doctor
```

## Basic Commands

### Initialize a New Project

```bash
# Create new MCP server project
fastmcp init my-server

# With template
fastmcp init my-server --template advanced

# Interactive setup
fastmcp init my-server --interactive
```

### Project Structure Created

```
my-server/
├── src/
│   └── server.py
├── tests/
│   └── test_server.py
├── fastmcp.json
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

### List Available Templates

```bash
# Show all templates
fastmcp templates

# Show template details
fastmcp templates --show basic
```

## Server Management

### Run Server in Production

```bash
# Run server
fastmcp run src/server.py

# With specific host and port
fastmcp run src/server.py --host 0.0.0.0 --port 8000

# With environment file
fastmcp run src/server.py --env-file .env.production

# With transport mode
fastmcp run src/server.py --transport stdio
fastmcp run src/server.py --transport websocket

# With workers (for websocket)
fastmcp run src/server.py --workers 4
```

### Development Server

```bash
# Run with auto-reload
fastmcp dev src/server.py

# With debug logging
fastmcp dev src/server.py --debug

# Watch specific files
fastmcp dev src/server.py --watch "*.py" --watch "*.json"

# With custom reload delay
fastmcp dev src/server.py --reload-delay 2
```

### Server Status and Management

```bash
# Check server status
fastmcp status

# List running servers
fastmcp list

# Stop server
fastmcp stop my-server
fastmcp stop --all

# Restart server
fastmcp restart my-server

# View server logs
fastmcp logs my-server
fastmcp logs my-server --follow
fastmcp logs my-server --tail 100
```

## Development Commands

### Generate Code

```bash
# Generate tool from OpenAPI spec
fastmcp generate tool --from-openapi api.yaml

# Generate resource
fastmcp generate resource --name user-profile --template dynamic

# Generate prompt
fastmcp generate prompt --name assistant --params "name,role"

# Generate complete server from spec
fastmcp generate server --from-spec spec.json
```

### Scaffold Components

```bash
# Scaffold new tool
fastmcp scaffold tool calculate_price

# Scaffold resource with template
fastmcp scaffold resource "user://{id}/profile"

# Scaffold middleware
fastmcp scaffold middleware auth_middleware

# Scaffold test
fastmcp scaffold test test_calculate_price
```

### Code Analysis

```bash
# Analyze server code
fastmcp analyze src/server.py

# Check for issues
fastmcp analyze src/server.py --check

# Generate report
fastmcp analyze src/server.py --report analysis.html

# Check complexity
fastmcp analyze src/server.py --complexity
```

## Configuration

### Initialize Configuration

```bash
# Create fastmcp.json
fastmcp config init

# Interactive configuration
fastmcp config init --interactive

# From existing server
fastmcp config init --from-server src/server.py
```

### View Configuration

```bash
# Show current config
fastmcp config show

# Show specific value
fastmcp config get server.name
fastmcp config get server.version

# List all settings
fastmcp config list
```

### Update Configuration

```bash
# Set configuration value
fastmcp config set server.name "my-awesome-server"
fastmcp config set server.version "2.0.0"
fastmcp config set features.cache true

# Add to arrays
fastmcp config add server.tags "production"
fastmcp config add server.tags "api"

# Remove from arrays
fastmcp config remove server.tags "test"
```

### Validate Configuration

```bash
# Validate fastmcp.json
fastmcp config validate

# Validate with schema
fastmcp config validate --schema strict

# Fix common issues
fastmcp config fix
```

## Testing Commands

### Run Tests

```bash
# Run all tests
fastmcp test

# Run specific test file
fastmcp test tests/test_tools.py

# Run specific test
fastmcp test tests/test_tools.py::test_calculate

# With coverage
fastmcp test --coverage

# With verbose output
fastmcp test -v

# Watch mode
fastmcp test --watch
```

### Test Server

```bash
# Test server endpoints
fastmcp test-server src/server.py

# Test specific tool
fastmcp test-server src/server.py --tool calculate

# Test resources
fastmcp test-server src/server.py --resources

# Load test
fastmcp test-server src/server.py --load --concurrent 10 --requests 1000
```

### Generate Test Data

```bash
# Generate test fixtures
fastmcp test-data generate

# From server schema
fastmcp test-data from-server src/server.py

# Specific number of items
fastmcp test-data generate --count 100

# Save to file
fastmcp test-data generate --output fixtures.json
```

## Deployment Commands

### Build for Deployment

```bash
# Build server package
fastmcp build

# With optimization
fastmcp build --optimize

# Specific output directory
fastmcp build --output dist/

# Include specific files
fastmcp build --include "*.json" --include "templates/"
```

### Deploy Server

```bash
# Deploy to cloud provider
fastmcp deploy --provider aws
fastmcp deploy --provider gcp
fastmcp deploy --provider azure

# Deploy to specific environment
fastmcp deploy --env production
fastmcp deploy --env staging

# With configuration
fastmcp deploy --config deploy.yaml

# Dry run
fastmcp deploy --dry-run
```

### Docker Support

```bash
# Generate Dockerfile
fastmcp docker init

# Build Docker image
fastmcp docker build --tag my-server:latest

# Run in Docker
fastmcp docker run my-server:latest

# Push to registry
fastmcp docker push my-server:latest --registry docker.io
```

## Utility Commands

### Validate Server

```bash
# Validate server implementation
fastmcp validate src/server.py

# Check tool signatures
fastmcp validate src/server.py --check-types

# Validate against MCP spec
fastmcp validate src/server.py --spec strict
```

### Format Code

```bash
# Format server code
fastmcp format src/

# Check formatting
fastmcp format src/ --check

# With specific style
fastmcp format src/ --style black
```

### Documentation

```bash
# Generate documentation
fastmcp docs generate

# From server
fastmcp docs from-server src/server.py

# Specific format
fastmcp docs generate --format markdown
fastmcp docs generate --format html

# Serve documentation
fastmcp docs serve --port 8080
```

### Server Info

```bash
# Show server information
fastmcp info src/server.py

# List tools
fastmcp info src/server.py --tools

# List resources
fastmcp info src/server.py --resources

# List prompts
fastmcp info src/server.py --prompts

# Export as JSON
fastmcp info src/server.py --json > server-info.json
```

## Environment Variables

### CLI Environment Variables

```bash
# Set FastMCP home directory
export FASTMCP_HOME=/opt/fastmcp

# Set default environment
export FASTMCP_ENV=production

# Enable debug mode
export FASTMCP_DEBUG=true

# Set log level
export FASTMCP_LOG_LEVEL=DEBUG

# Config file location
export FASTMCP_CONFIG=/etc/fastmcp/config.json

# Disable telemetry
export FASTMCP_TELEMETRY=false
```

### Server Environment Variables

```bash
# Server configuration
export MCP_SERVER_NAME="my-server"
export MCP_SERVER_VERSION="1.0.0"
export MCP_SERVER_HOST="0.0.0.0"
export MCP_SERVER_PORT="8000"

# Transport settings
export MCP_TRANSPORT="stdio"  # or "websocket"

# Feature flags
export MCP_ENABLE_CACHE="true"
export MCP_ENABLE_AUTH="true"
export MCP_ENABLE_METRICS="true"

# API configuration
export MCP_API_BASE_URL="https://api.example.com"
export MCP_API_KEY="your-api-key"
export MCP_API_TIMEOUT="30"
```

## Configuration Files

### fastmcp.json

```json
{
  "name": "my-mcp-server",
  "version": "1.0.0",
  "description": "My awesome MCP server",
  "author": "Your Name",
  "license": "MIT",
  
  "server": {
    "entry": "src/server.py",
    "host": "localhost",
    "port": 8000,
    "transport": "stdio",
    "workers": 1
  },
  
  "features": {
    "cache": true,
    "auth": false,
    "metrics": true,
    "logging": {
      "level": "INFO",
      "format": "json",
      "file": "server.log"
    }
  },
  
  "dependencies": {
    "fastmcp": "^0.2.0",
    "aiohttp": "^3.9.0",
    "pydantic": "^2.0.0"
  },
  
  "scripts": {
    "start": "fastmcp run src/server.py",
    "dev": "fastmcp dev src/server.py --debug",
    "test": "fastmcp test",
    "build": "fastmcp build --optimize"
  },
  
  "tools": {
    "include": ["src/tools/*.py"],
    "exclude": ["src/tools/experimental.py"]
  },
  
  "resources": {
    "static": "resources/static",
    "templates": "resources/templates"
  },
  
  "deployment": {
    "provider": "aws",
    "region": "us-west-2",
    "environment": {
      "production": {
        "instances": 3,
        "memory": 512,
        "timeout": 30
      },
      "staging": {
        "instances": 1,
        "memory": 256,
        "timeout": 60
      }
    }
  }
}
```

### .fastmcp.yaml

```yaml
# Alternative YAML configuration
name: my-mcp-server
version: 1.0.0
description: My awesome MCP server

server:
  entry: src/server.py
  host: localhost
  port: 8000
  transport: stdio
  
features:
  cache: true
  auth: false
  metrics: true
  logging:
    level: INFO
    format: json
    file: server.log

scripts:
  start: fastmcp run src/server.py
  dev: fastmcp dev src/server.py --debug
  test: fastmcp test
  build: fastmcp build --optimize

environments:
  development:
    server:
      host: localhost
      port: 8001
      debug: true
  
  production:
    server:
      host: 0.0.0.0
      port: 80
      workers: 4
    features:
      auth: true
      cache: true
```

### pyproject.toml Integration

```toml
[tool.fastmcp]
name = "my-mcp-server"
version = "1.0.0"

[tool.fastmcp.server]
entry = "src/server.py"
transport = "stdio"

[tool.fastmcp.scripts]
start = "fastmcp run src/server.py"
dev = "fastmcp dev src/server.py --debug"
test = "fastmcp test"

[tool.fastmcp.features]
cache = true
auth = false
metrics = true
```

## Advanced Usage

### Custom Commands

```bash
# Register custom command
fastmcp plugin add my-plugin

# Run custom command
fastmcp my-command --option value

# List plugins
fastmcp plugin list
```

### Batch Operations

```bash
# Run multiple servers
fastmcp batch run server1.py server2.py server3.py

# Test multiple servers
fastmcp batch test tests/*.py

# Deploy multiple environments
fastmcp batch deploy --envs "staging,production"
```

### Pipeline Commands

```bash
# Run complete pipeline
fastmcp pipeline run

# Define pipeline
cat > pipeline.yaml << EOF
stages:
  - name: test
    command: fastmcp test
  - name: build
    command: fastmcp build
  - name: deploy
    command: fastmcp deploy --env staging
EOF

fastmcp pipeline run --file pipeline.yaml
```

### Integration Commands

```bash
# Integrate with CI/CD
fastmcp ci setup github-actions
fastmcp ci setup gitlab-ci
fastmcp ci setup jenkins

# Generate CI configuration
fastmcp ci generate --provider github

# Validate CI configuration
fastmcp ci validate .github/workflows/test.yml
```

## Troubleshooting

### Debug Mode

```bash
# Run with debug output
fastmcp --debug run src/server.py

# Verbose logging
fastmcp -vvv run src/server.py

# Trace mode
FASTMCP_TRACE=1 fastmcp run src/server.py
```

### Doctor Command

```bash
# Check system health
fastmcp doctor

# Detailed diagnostics
fastmcp doctor --verbose

# Fix common issues
fastmcp doctor --fix

# Check specific component
fastmcp doctor --check environment
fastmcp doctor --check dependencies
fastmcp doctor --check configuration
```

### Common Issues

```bash
# Clear cache
fastmcp cache clear

# Reset configuration
fastmcp config reset

# Reinstall dependencies
fastmcp deps install --force

# Update FastMCP
fastmcp self-update
```

### Log Management

```bash
# View logs
fastmcp logs

# Clear logs
fastmcp logs clear

# Export logs
fastmcp logs export --format json > logs.json

# Analyze logs
fastmcp logs analyze --errors
fastmcp logs analyze --performance
```

## Command Reference

### Global Options

```bash
fastmcp [OPTIONS] COMMAND [ARGS]...

Options:
  --version           Show version
  --help             Show help message
  -v, --verbose      Increase verbosity
  -q, --quiet        Suppress output
  --debug            Enable debug mode
  --config PATH      Config file path
  --env-file PATH    Environment file
  --no-color         Disable colored output
  --json             JSON output format
```

### Complete Command List

```bash
# Server Management
fastmcp run            # Run server
fastmcp dev            # Development server
fastmcp stop           # Stop server
fastmcp restart        # Restart server
fastmcp status         # Server status
fastmcp list           # List servers

# Development
fastmcp init           # Initialize project
fastmcp generate       # Generate code
fastmcp scaffold       # Scaffold components
fastmcp analyze        # Analyze code
fastmcp format         # Format code
fastmcp validate       # Validate server

# Configuration
fastmcp config init    # Initialize config
fastmcp config show    # Show config
fastmcp config set     # Set config value
fastmcp config get     # Get config value
fastmcp config validate # Validate config

# Testing
fastmcp test           # Run tests
fastmcp test-server    # Test server
fastmcp test-data      # Generate test data

# Deployment
fastmcp build          # Build package
fastmcp deploy         # Deploy server
fastmcp docker         # Docker commands

# Documentation
fastmcp docs generate  # Generate docs
fastmcp docs serve     # Serve docs
fastmcp info          # Server info

# Utilities
fastmcp doctor         # System diagnostics
fastmcp cache          # Cache management
fastmcp logs           # Log management
fastmcp templates      # List templates
fastmcp plugin         # Plugin management
```

## Examples

### Complete Development Workflow

```bash
# 1. Create new project
fastmcp init weather-server --template advanced

# 2. Navigate to project
cd weather-server

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure server
fastmcp config set server.name "weather-mcp"
fastmcp config set features.cache true

# 5. Run development server
fastmcp dev src/server.py --debug

# 6. Run tests
fastmcp test --coverage

# 7. Build for production
fastmcp build --optimize

# 8. Deploy
fastmcp deploy --env production
```

### Testing Workflow

```bash
# 1. Generate test data
fastmcp test-data generate --count 100 --output fixtures.json

# 2. Run unit tests
fastmcp test tests/unit/

# 3. Run integration tests
fastmcp test tests/integration/ --slow

# 4. Load test server
fastmcp test-server src/server.py --load --concurrent 50

# 5. Generate coverage report
fastmcp test --coverage --report html
```

### Production Deployment

```bash
# 1. Validate server
fastmcp validate src/server.py --spec strict

# 2. Run tests
fastmcp test --all

# 3. Build optimized package
fastmcp build --optimize --output dist/

# 4. Generate Docker image
fastmcp docker build --tag weather-server:v1.0.0

# 5. Deploy to cloud
fastmcp deploy \
  --provider aws \
  --env production \
  --config deploy.yaml \
  --confirm

# 6. Monitor deployment
fastmcp status --follow
```

## Summary

The FastMCP CLI provides a complete toolkit for MCP server development:

1. **Development**: Easy project initialization and development servers
2. **Testing**: Comprehensive testing utilities and fixtures
3. **Configuration**: Flexible configuration management
4. **Deployment**: Built-in deployment to major cloud providers
5. **Management**: Server lifecycle management and monitoring
6. **Utilities**: Code generation, formatting, and documentation
7. **Integration**: CI/CD pipeline support

Key features:
- Auto-reload development server
- Interactive project setup
- Code generation from OpenAPI
- Built-in testing framework
- Docker support
- Cloud deployment
- Comprehensive logging
- Plugin system

The CLI is designed to support the complete development lifecycle from project creation to production deployment.