"""
FastMCP Server Template
=======================
A production-ready template for FastMCP servers with best practices.
Customize this template for your specific use case.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Application configuration from environment variables."""
    
    # Server
    SERVER_NAME = os.getenv("SERVER_NAME", "My FastMCP Server")
    SERVER_VERSION = "1.0.0"
    
    # API Configuration (if needed)
    API_BASE_URL = os.getenv("API_BASE_URL", "")
    API_KEY = os.getenv("API_KEY", "")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Add your configuration here
    # CUSTOM_SETTING = os.getenv("CUSTOM_SETTING", "default_value")


# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Server Creation
# ============================================================================

def create_server() -> FastMCP:
    """
    Factory function to create the MCP server.
    This pattern allows for complex initialization.
    """
    
    mcp = FastMCP(
        name=Config.SERVER_NAME,
        instructions=f"""
        {Config.SERVER_NAME} v{Config.SERVER_VERSION}
        
        This server provides [describe your functionality here].
        
        Available tools:
        - example_tool: Demonstrates tool creation
        - process_data: Process input data
        
        Available resources:
        - info://status: Server status
        - data://config: Configuration information
        """
    )
    
    # ========== Tools ==========
    
    @mcp.tool
    def example_tool(
        input_text: str,
        option: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Example tool demonstrating best practices.
        
        Args:
            input_text: The text to process
            option: Optional processing option
        
        Returns:
            Processed result with metadata
        """
        # Input validation
        if not input_text:
            return {"error": "Input text is required"}
        
        # Process the input
        result = input_text.upper() if option == "uppercase" else input_text
        
        # Return structured response
        return {
            "success": True,
            "original": input_text,
            "processed": result,
            "option_used": option,
            "timestamp": datetime.now().isoformat()
        }
    
    @mcp.tool
    async def process_data(
        data: List[Dict[str, Any]],
        operation: str = "summarize"
    ) -> Dict[str, Any]:
        """
        Process a list of data items.
        
        Args:
            data: List of data items to process
            operation: Operation to perform (summarize, filter, transform)
        
        Returns:
            Processed data with operation results
        """
        try:
            if operation == "summarize":
                return {
                    "count": len(data),
                    "first": data[0] if data else None,
                    "last": data[-1] if data else None
                }
            elif operation == "filter":
                # Example: filter items with 'active' flag
                filtered = [d for d in data if d.get("active", False)]
                return {
                    "original_count": len(data),
                    "filtered_count": len(filtered),
                    "filtered_data": filtered
                }
            elif operation == "transform":
                # Example transformation
                transformed = [
                    {**d, "processed_at": datetime.now().isoformat()}
                    for d in data
                ]
                return {"transformed_data": transformed}
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return {"error": str(e)}
    
    # Add more tools as needed
    # @mcp.tool
    # def your_custom_tool():
    #     pass
    
    # ========== Resources ==========
    
    @mcp.resource("info://status")
    def server_status() -> Dict[str, Any]:
        """Get current server status."""
        return {
            "server": Config.SERVER_NAME,
            "version": Config.SERVER_VERSION,
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "api_configured": bool(Config.API_KEY)
        }
    
    @mcp.resource("data://config")
    def server_config() -> Dict[str, Any]:
        """Get server configuration (non-sensitive)."""
        return {
            "server_name": Config.SERVER_NAME,
            "version": Config.SERVER_VERSION,
            "log_level": Config.LOG_LEVEL,
            "api_base_url": Config.API_BASE_URL if Config.API_BASE_URL else "Not configured"
        }
    
    # Resource template example
    @mcp.resource("data://items/{item_id}")
    def get_item(item_id: str) -> Dict[str, Any]:
        """Get specific item by ID."""
        # This would typically fetch from a database or API
        return {
            "id": item_id,
            "name": f"Item {item_id}",
            "retrieved_at": datetime.now().isoformat()
        }
    
    # ========== Prompts ==========
    
    @mcp.prompt("help")
    def help_prompt() -> str:
        """Generate help prompt for the server."""
        return f"""
        Welcome to {Config.SERVER_NAME}!
        
        You can use the following tools:
        1. example_tool - Process text with optional transformations
        2. process_data - Process lists of data with various operations
        
        Available resources:
        - info://status - Check server status
        - data://config - View configuration
        - data://items/{{id}} - Get specific items
        
        Example usage:
        - Call example_tool with text to process it
        - Use process_data to summarize, filter, or transform data lists
        """
    
    @mcp.prompt("analyze")
    def analyze_prompt(topic: str) -> str:
        """Generate analysis prompt for a topic."""
        return f"""
        Please analyze the following topic: {topic}
        
        Consider:
        1. Current state and context
        2. Key challenges or issues
        3. Potential opportunities
        4. Recommended actions
        5. Expected outcomes
        
        Use the available tools to gather and process relevant data.
        """
    
    logger.info(f"Server '{Config.SERVER_NAME}' created successfully")
    return mcp


# ============================================================================
# Optional: API Integration
# ============================================================================

def add_api_tools(mcp: FastMCP):
    """
    Add API integration tools if API is configured.
    This is separated for clarity and modularity.
    """
    if not Config.API_KEY:
        logger.info("API not configured, skipping API tools")
        return
    
    import httpx
    
    @mcp.tool
    async def api_request(
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the configured API.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Request body data for POST/PUT
        
        Returns:
            API response or error
        """
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {Config.API_KEY}"}
                url = f"{Config.API_BASE_URL}{endpoint}"
                
                if method == "GET":
                    response = await client.get(url, headers=headers)
                elif method == "POST":
                    response = await client.post(url, json=data, headers=headers)
                elif method == "PUT":
                    response = await client.put(url, json=data, headers=headers)
                elif method == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    return {"error": f"Unsupported method: {method}"}
                
                response.raise_for_status()
                return {
                    "success": True,
                    "data": response.json(),
                    "status_code": response.status_code
                }
                
        except httpx.HTTPStatusError as e:
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "message": e.response.text
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    logger.info("API tools added")


# ============================================================================
# Main Execution
# ============================================================================

# Create server instance (for module import)
mcp = create_server()

# Add optional API tools
add_api_tools(mcp)


def main():
    """Main entry point for direct execution."""
    import sys
    
    # Check for test mode
    if "--test" in sys.argv:
        logger.info("Running in test mode...")
        # Add test code here
        import asyncio
        from fastmcp import Client
        
        async def test():
            async with Client(mcp) as client:
                tools = await client.list_tools()
                print(f"Available tools: {[t.name for t in tools]}")
                
                result = await client.call_tool(
                    "example_tool",
                    {"input_text": "Hello, World!", "option": "uppercase"}
                )
                print(f"Tool result: {result.data}")
        
        asyncio.run(test())
        return
    
    # Parse transport option
    transport = "stdio"  # default
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        if idx + 1 < len(sys.argv):
            transport = sys.argv[idx + 1]
    
    logger.info(f"Starting server with transport: {transport}")
    
    # Run server
    if transport == "http":
        port = 8000
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        
        mcp.run(transport="http", port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        exit(1)