#!/usr/bin/env python3
"""
Modular FastMCP Server
======================
Production-ready MCP server with modular architecture.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for shared module access
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from shared import Config, format_success, format_error

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name=Config.SERVER_NAME,
    version=Config.SERVER_VERSION
)

# ============================================================
# Import and register tools
# ============================================================

from tools import (
    # Data tools
    process_data,
    transform_data,
    validate_data,
    export_data,
    
    # API tools
    fetch_api_data,
    post_api_data,
    update_api_resource,
    delete_api_resource,
    
    # File tools
    read_file,
    write_file,
    list_files,
    process_csv,
    
    # Utility tools
    health_check,
    get_status,
    run_diagnostics
)

# Register all tools
tools_to_register = [
    # Data processing
    process_data,
    transform_data,
    validate_data,
    export_data,
    
    # API operations
    fetch_api_data,
    post_api_data,
    update_api_resource,
    delete_api_resource,
    
    # File operations
    read_file,
    write_file,
    list_files,
    process_csv,
    
    # Utilities
    health_check,
    get_status,
    run_diagnostics
]

for tool in tools_to_register:
    mcp.tool(tool)
    logger.debug(f"Registered tool: {tool.__name__}")

# ============================================================
# Import and register resources
# ============================================================

from resources import (
    # Static resources
    server_info,
    configuration,
    
    # Dynamic resources (templates)
    get_user_profile,
    get_project_data,
    get_api_endpoint
)

# Register static resources
mcp.resource("info://server")(server_info)
mcp.resource("config://settings")(configuration)

# Register resource templates
mcp.resource("user://{user_id}/profile")(get_user_profile)
mcp.resource("project://{project_id}/data")(get_project_data)
mcp.resource("api://{version}/{endpoint}")(get_api_endpoint)

logger.debug(f"Registered {len(mcp.resources)} resources")

# ============================================================
# Import and register prompts
# ============================================================

from prompts import (
    analysis_prompt,
    summary_prompt,
    debug_prompt,
    optimization_prompt
)

# Register prompts
mcp.prompt("analyze")(analysis_prompt)
mcp.prompt("summarize")(summary_prompt)
mcp.prompt("debug")(debug_prompt)
mcp.prompt("optimize")(optimization_prompt)

logger.debug(f"Registered {len(mcp.prompts)} prompts")

# ============================================================
# Advanced features (if enabled)
# ============================================================

if Config.ENABLE_ADVANCED_FEATURES:
    try:
        from tools.advanced import (
            interactive_setup,
            batch_processor,
            ai_assistant
        )
        
        # Register advanced tools
        mcp.tool(interactive_setup)
        mcp.tool(batch_processor)
        mcp.tool(ai_assistant)
        
        logger.info("Advanced features enabled")
    except ImportError as e:
        logger.warning(f"Advanced features not available: {e}")

# ============================================================
# Server lifecycle
# ============================================================

@mcp.on_startup
async def startup():
    """Initialize server resources on startup."""
    logger.info(f"Starting {Config.SERVER_NAME} v{Config.SERVER_VERSION}")
    
    # Initialize shared resources
    from shared import initialize_cache, verify_api_connection
    
    await initialize_cache()
    
    if Config.API_BASE_URL:
        if await verify_api_connection():
            logger.info("API connection verified")
        else:
            logger.warning("API connection failed - some features may be unavailable")

@mcp.on_shutdown
async def shutdown():
    """Cleanup on server shutdown."""
    logger.info("Shutting down server")
    
    # Cleanup shared resources
    from shared import cleanup_cache, close_api_client
    
    await cleanup_cache()
    await close_api_client()

# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Modular FastMCP Server")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        # Test mode - verify all components
        print(f"Server: {Config.SERVER_NAME} v{Config.SERVER_VERSION}")
        print(f"Tools: {len(tools_to_register)}")
        print(f"Resources: {len(mcp.resources)}")
        print(f"Prompts: {len(mcp.prompts)}")
        print(f"Environment: {Config.ENVIRONMENT}")
        print("All components loaded successfully!")
    else:
        # Run server
        logger.info(f"Starting server on {Config.TRANSPORT}")
        mcp.run()