#!/usr/bin/env python3
"""
Advanced FastMCP Server with All v2 Features
Demonstrates resource templates, elicitation, progress tracking, sampling, and more.
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_advanced_server() -> FastMCP:
    """
    Factory function to create an advanced MCP server with all features.
    """
    mcp = FastMCP(
        name=os.getenv("SERVER_NAME", "Advanced FastMCP Server"),
        version="2.0.0"
    )
    
    # ============================================================
    # RESOURCE TEMPLATES - Dynamic resources with parameters
    # ============================================================
    
    @mcp.resource("user://{user_id}/profile")
    async def get_user_profile(user_id: str) -> Dict[str, Any]:
        """Get user profile by ID."""
        # Simulate database lookup
        await asyncio.sleep(0.1)
        return {
            "id": user_id,
            "name": f"User {user_id}",
            "created": datetime.now().isoformat(),
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
    
    @mcp.resource("api://{version}/status")
    async def get_api_status(version: str) -> Dict[str, Any]:
        """Get API status for specific version."""
        return {
            "version": version,
            "status": "healthy",
            "uptime": "99.9%",
            "endpoints": {
                "v1": ["users", "posts", "comments"],
                "v2": ["users", "posts", "comments", "reactions"],
                "v3": ["users", "posts", "comments", "reactions", "analytics"]
            }.get(version, [])
        }
    
    @mcp.resource("data://{category}/{item_id}")
    async def get_categorized_data(category: str, item_id: str) -> Dict[str, Any]:
        """Get data by category and item ID."""
        return {
            "category": category,
            "item_id": item_id,
            "data": f"Content for {category}/{item_id}",
            "metadata": {
                "accessed": datetime.now().isoformat(),
                "size": len(f"Content for {category}/{item_id}")
            }
        }
    
    # ============================================================
    # ELICITATION - Interactive user input during execution
    # ============================================================
    
    @mcp.tool
    async def interactive_setup(context) -> Dict[str, Any]:
        """
        Interactive setup wizard using elicitation.
        Demonstrates different input types and validation.
        """
        # Get project name
        project_name = await context.elicit(
            "What is your project name?",
            str,
            {"context": "setup", "step": 1}
        )
        
        # Get project type with validation
        project_type = None
        while project_type not in ["api", "web", "cli", "library"]:
            project_type = await context.elicit(
                "Project type? (api/web/cli/library)",
                str,
                {"context": "setup", "step": 2}
            )
            if project_type not in ["api", "web", "cli", "library"]:
                logger.warning(f"Invalid project type: {project_type}")
        
        # Get port number
        port_str = await context.elicit(
            "Port number? (default: 8000)",
            str,
            {"context": "setup", "step": 3, "default": "8000"}
        )
        port = int(port_str) if port_str else 8000
        
        # Confirm settings
        confirm = await context.elicit(
            f"Create {project_type} project '{project_name}' on port {port}? (yes/no)",
            str,
            {"context": "setup", "step": 4}
        )
        
        if confirm.lower() != "yes":
            return {"status": "cancelled"}
        
        return {
            "status": "created",
            "project": {
                "name": project_name,
                "type": project_type,
                "port": port,
                "created": datetime.now().isoformat()
            }
        }
    
    @mcp.tool
    async def secure_operation(context) -> Dict[str, Any]:
        """
        Demonstrates password/secret elicitation.
        """
        # Get password (should be masked in UI)
        password = await context.elicit(
            "Enter admin password:",
            str,
            {"context": "auth", "sensitive": True}
        )
        
        # Simulate verification
        if len(password) < 8:
            return {"error": "Password too short"}
        
        # Get 2FA code
        code = await context.elicit(
            "Enter 2FA code:",
            str,
            {"context": "auth", "type": "otp"}
        )
        
        if len(code) != 6 or not code.isdigit():
            return {"error": "Invalid 2FA code"}
        
        return {
            "status": "authenticated",
            "session": "token-" + datetime.now().strftime("%Y%m%d%H%M%S")
        }
    
    # ============================================================
    # PROGRESS TRACKING - Monitor long-running operations
    # ============================================================
    
    @mcp.tool
    async def process_batch(context, items: List[str]) -> Dict[str, Any]:
        """
        Process items with progress reporting.
        """
        results = []
        total = len(items)
        
        for i, item in enumerate(items):
            # Report progress
            await context.report_progress(
                i + 1,
                total,
                f"Processing {item}"
            )
            
            # Simulate processing
            await asyncio.sleep(0.5)
            results.append(f"Processed: {item}")
            
            # Report sub-progress for complex items
            if "complex" in item.lower():
                for j in range(5):
                    await context.report_progress(
                        i + (j / 5),
                        total,
                        f"Complex processing step {j+1}/5 for {item}"
                    )
                    await asyncio.sleep(0.1)
        
        await context.report_progress(total, total, "Batch processing complete")
        
        return {
            "status": "complete",
            "processed": len(results),
            "results": results
        }
    
    @mcp.tool
    async def download_files(context, urls: List[str]) -> Dict[str, Any]:
        """
        Download files with detailed progress.
        """
        downloads = []
        
        for i, url in enumerate(urls):
            file_name = url.split("/")[-1]
            
            # Simulate download with incremental progress
            file_size = 1024 * 1024  # 1MB simulated
            downloaded = 0
            chunk_size = 102400  # 100KB chunks
            
            while downloaded < file_size:
                downloaded += chunk_size
                if downloaded > file_size:
                    downloaded = file_size
                
                # Report download progress
                await context.report_progress(
                    downloaded,
                    file_size,
                    f"Downloading {file_name}: {downloaded/1024:.1f}KB / {file_size/1024:.1f}KB"
                )
                await asyncio.sleep(0.1)
            
            downloads.append({
                "url": url,
                "file": file_name,
                "size": file_size
            })
        
        return {
            "status": "complete",
            "downloads": downloads,
            "total_size": sum(d["size"] for d in downloads)
        }
    
    # ============================================================
    # SAMPLING - LLM integration for intelligent responses
    # ============================================================
    
    @mcp.tool
    async def generate_code(
        context,
        language: str,
        description: str,
        style: str = "clean"
    ) -> Dict[str, Any]:
        """
        Generate code using LLM sampling.
        """
        # Build prompt
        messages = [
            {
                "role": "system",
                "content": f"You are a {language} code generator. Generate {style} code."
            },
            {
                "role": "user",
                "content": f"Generate {language} code for: {description}"
            }
        ]
        
        # Request LLM generation
        result = await context.sample(
            messages=messages,
            model="gpt-4",
            temperature=0.7,
            max_tokens=500
        )
        
        return {
            "language": language,
            "description": description,
            "code": result.get("content", ""),
            "model": result.get("model", "unknown"),
            "tokens": result.get("usage", {}).get("tokens", 0)
        }
    
    @mcp.tool
    async def analyze_text(context, text: str) -> Dict[str, Any]:
        """
        Analyze text using LLM with specific instructions.
        """
        # Multi-step analysis with sampling
        
        # Step 1: Sentiment analysis
        sentiment_result = await context.sample(
            messages=[
                {"role": "system", "content": "Analyze sentiment. Reply with only: positive, negative, or neutral"},
                {"role": "user", "content": text}
            ],
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=10
        )
        
        # Step 2: Key topics
        topics_result = await context.sample(
            messages=[
                {"role": "system", "content": "Extract 3 key topics as comma-separated list"},
                {"role": "user", "content": text}
            ],
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=50
        )
        
        # Step 3: Summary
        summary_result = await context.sample(
            messages=[
                {"role": "system", "content": "Summarize in one sentence"},
                {"role": "user", "content": text}
            ],
            model="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=100
        )
        
        return {
            "original_length": len(text),
            "sentiment": sentiment_result.get("content", "unknown"),
            "topics": topics_result.get("content", "").split(","),
            "summary": summary_result.get("content", ""),
            "analysis_complete": datetime.now().isoformat()
        }
    
    @mcp.tool
    async def smart_search(
        ctx,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Smart search with LLM-enhanced query understanding.
        """
        # First, understand the query intent
        intent_result = await ctx.sample(
            messages=[
                {
                    "role": "system",
                    "content": "Classify search intent: informational, navigational, transactional, or local"
                },
                {"role": "user", "content": query}
            ],
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=20
        )
        
        intent = intent_result.get("content", "informational")
        
        # Expand query with synonyms and related terms
        expansion_result = await ctx.sample(
            messages=[
                {
                    "role": "system",
                    "content": "Generate 3 related search terms, comma-separated"
                },
                {"role": "user", "content": f"Query: {query}\nContext: {context or 'general'}"}
            ],
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=50
        )
        
        related_terms = expansion_result.get("content", "").split(",")
        
        # Simulate search with enhanced query
        search_results = [
            {"title": f"Result for {query}", "relevance": 0.95},
            {"title": f"Related to {related_terms[0] if related_terms else query}", "relevance": 0.80},
            {"title": f"Context: {context or 'general'}", "relevance": 0.75}
        ]
        
        return {
            "query": query,
            "intent": intent,
            "expanded_terms": related_terms,
            "results": search_results,
            "context": context
        }
    
    # ============================================================
    # COMBINED FEATURES - Using multiple v2 features together
    # ============================================================
    
    @mcp.tool
    async def intelligent_pipeline(context, data_source: str) -> Dict[str, Any]:
        """
        Demonstrates combining elicitation, progress, and sampling.
        """
        # Step 1: Get processing preferences via elicitation
        await context.report_progress(0, 5, "Getting processing preferences")
        
        mode = await context.elicit(
            "Processing mode? (fast/accurate/balanced)",
            str,
            {"step": "configuration"}
        )
        
        # Step 2: Analyze data source with LLM
        await context.report_progress(1, 5, "Analyzing data source")
        
        analysis = await context.sample(
            messages=[
                {"role": "system", "content": "Analyze data source and suggest processing approach"},
                {"role": "user", "content": f"Data source: {data_source}, Mode: {mode}"}
            ],
            model="gpt-4",
            temperature=0.3
        )
        
        # Step 3: Process with progress tracking
        await context.report_progress(2, 5, "Processing data")
        await asyncio.sleep(1)  # Simulate processing
        
        # Step 4: Get user validation
        await context.report_progress(3, 5, "Awaiting validation")
        
        validation = await context.elicit(
            f"Processing suggestion: {analysis.get('content', 'Standard processing')}. Proceed? (yes/no)",
            str,
            {"step": "validation"}
        )
        
        if validation.lower() != "yes":
            return {"status": "cancelled"}
        
        # Step 5: Final processing
        await context.report_progress(4, 5, "Finalizing")
        await asyncio.sleep(0.5)
        
        await context.report_progress(5, 5, "Complete")
        
        return {
            "status": "complete",
            "source": data_source,
            "mode": mode,
            "analysis": analysis.get("content"),
            "processed": datetime.now().isoformat()
        }
    
    # ============================================================
    # STATIC RESOURCES - Traditional resources for comparison
    # ============================================================
    
    @mcp.resource("info://server")
    def server_info() -> Dict[str, Any]:
        """Server information and capabilities."""
        return {
            "name": mcp.name,
            "version": "2.0.0",
            "features": [
                "resource_templates",
                "elicitation",
                "progress_tracking",
                "sampling",
                "async_operations"
            ],
            "status": "ready"
        }
    
    # ============================================================
    # PROMPTS - Reusable prompt templates
    # ============================================================
    
    @mcp.prompt("debug")
    def debug_prompt(component: str, error: Optional[str] = None) -> str:
        """Generate debugging prompt."""
        base = f"Debug the {component} component."
        if error:
            base += f" Error: {error}"
        base += " Provide step-by-step troubleshooting."
        return base
    
    @mcp.prompt("optimize")
    def optimize_prompt(code: str, target: str = "performance") -> str:
        """Generate optimization prompt."""
        return f"""
        Optimize the following code for {target}:
        
        ```
        {code}
        ```
        
        Provide optimized version with explanations.
        """
    
    # ============================================================
    # HEALTH CHECK - Always useful
    # ============================================================
    
    @mcp.tool
    def health_check() -> Dict[str, Any]:
        """Comprehensive health check."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "resource_templates": True,
                "elicitation": True,
                "progress_tracking": True,
                "sampling": True
            },
            "environment": {
                "has_api_key": bool(os.getenv("API_KEY")),
                "log_level": os.getenv("LOG_LEVEL", "INFO")
            }
        }
    
    logger.info(f"Advanced server initialized: {mcp.name}")
    return mcp


# Create and run server
mcp = create_advanced_server()

if __name__ == "__main__":
    import sys
    
    if "--test" in sys.argv:
        # Test mode
        print("Running in test mode...")
        print(f"Server: {mcp.name}")
        print(f"Tools: {len(mcp.tools)}")
        print(f"Resources: {len(mcp.resources)}")
        print("Health check:", mcp.tools["health_check"]())
    else:
        # Run server
        mcp.run()