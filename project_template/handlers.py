"""
Client Handlers for FastMCP
Implements elicitation, progress, and sampling handlers for MCP clients.
"""

import os
import sys
import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

# For actual LLM integration
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class ElicitationHandler:
    """
    Handle elicitation requests from MCP server.
    Provides different strategies for getting user input.
    """
    
    def __init__(self, mode: str = "interactive"):
        """
        Initialize elicitation handler.
        
        Args:
            mode: "interactive", "auto", "gui", or "cli"
        """
        self.mode = mode
        self.history = []
    
    async def handle(
        self,
        message: str,
        response_type: type,
        context: Dict[str, Any]
    ) -> Any:
        """
        Handle elicitation request.
        
        Args:
            message: The prompt message
            response_type: Expected response type (str, int, bool, etc.)
            context: Additional context from server
        
        Returns:
            User response of appropriate type
        """
        # Log request
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "context": context
        })
        
        if self.mode == "interactive":
            return await self._interactive_input(message, response_type, context)
        elif self.mode == "auto":
            return await self._auto_response(message, response_type, context)
        elif self.mode == "gui":
            return await self._gui_input(message, response_type, context)
        else:  # cli
            return await self._cli_input(message, response_type, context)
    
    async def _interactive_input(
        self,
        message: str,
        response_type: type,
        context: Dict[str, Any]
    ) -> Any:
        """Interactive console input."""
        # Check if sensitive (password, token, etc.)
        sensitive = context.get("sensitive", False)
        
        if sensitive:
            # Use getpass for sensitive input
            import getpass
            response = getpass.getpass(f"{message}: ")
        else:
            # Regular input
            response = input(f"{message}: ")
        
        # Convert to appropriate type
        if response_type == int:
            return int(response)
        elif response_type == float:
            return float(response)
        elif response_type == bool:
            return response.lower() in ["yes", "true", "1", "y"]
        else:
            return response
    
    async def _auto_response(
        self,
        message: str,
        response_type: type,
        context: Dict[str, Any]
    ) -> Any:
        """Automatic response based on context."""
        # Use defaults if provided
        if "default" in context:
            return context["default"]
        
        # Generate appropriate response
        if response_type == bool:
            return True  # Default to yes
        elif response_type == int:
            return 0
        elif response_type == float:
            return 0.0
        else:
            return f"auto-response-{datetime.now().timestamp()}"
    
    async def _gui_input(
        self,
        message: str,
        response_type: type,
        context: Dict[str, Any]
    ) -> Any:
        """GUI input using tkinter (if available)."""
        try:
            import tkinter as tk
            from tkinter import simpledialog, messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            if context.get("sensitive"):
                # Password dialog
                response = simpledialog.askstring(
                    "Secure Input",
                    message,
                    show="*"
                )
            elif response_type == bool:
                # Yes/No dialog
                response = messagebox.askyesno("Confirmation", message)
            else:
                # Text input dialog
                response = simpledialog.askstring("Input Required", message)
            
            root.destroy()
            
            # Convert type
            if response_type == int:
                return int(response)
            elif response_type == float:
                return float(response)
            else:
                return response
                
        except ImportError:
            # Fallback to CLI
            logger.warning("tkinter not available, falling back to CLI input")
            return await self._cli_input(message, response_type, context)
    
    async def _cli_input(
        self,
        message: str,
        response_type: type,
        context: Dict[str, Any]
    ) -> Any:
        """CLI input with validation."""
        while True:
            try:
                response = input(f"\n[{context.get('step', 'input')}] {message}: ")
                
                # Validate and convert
                if response_type == int:
                    return int(response)
                elif response_type == float:
                    return float(response)
                elif response_type == bool:
                    if response.lower() in ["yes", "no", "true", "false", "y", "n"]:
                        return response.lower() in ["yes", "true", "y"]
                    else:
                        print("Please enter yes/no")
                        continue
                else:
                    return response
                    
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")
            except KeyboardInterrupt:
                print("\nInput cancelled")
                return None


class ProgressHandler:
    """
    Handle progress updates from MCP server.
    Provides different visualization strategies.
    """
    
    def __init__(self, mode: str = "bar"):
        """
        Initialize progress handler.
        
        Args:
            mode: "bar", "percentage", "dots", "silent"
        """
        self.mode = mode
        self.last_progress = {}
        
        # For progress bar
        self.bar_width = 50
        self.last_update = datetime.now()
    
    async def handle(
        self,
        progress: float,
        total: Optional[float],
        message: Optional[str]
    ) -> None:
        """
        Handle progress update.
        
        Args:
            progress: Current progress value
            total: Total value (None for indeterminate)
            message: Progress message
        """
        if self.mode == "silent":
            return
        elif self.mode == "bar":
            await self._show_bar(progress, total, message)
        elif self.mode == "percentage":
            await self._show_percentage(progress, total, message)
        elif self.mode == "dots":
            await self._show_dots(progress, total, message)
    
    async def _show_bar(
        self,
        progress: float,
        total: Optional[float],
        message: Optional[str]
    ) -> None:
        """Show progress bar."""
        if total:
            # Calculate percentage
            percentage = (progress / total) * 100
            filled = int(self.bar_width * progress / total)
            
            # Create bar
            bar = "█" * filled + "░" * (self.bar_width - filled)
            
            # Print with carriage return for updating
            output = f"\r[{bar}] {percentage:.1f}%"
            if message:
                output += f" - {message}"
            
            print(output, end="", flush=True)
            
            # New line when complete
            if progress >= total:
                print()
        else:
            # Indeterminate progress
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            idx = int((datetime.now().timestamp() * 10) % len(spinner))
            
            output = f"\r{spinner[idx]} {message or 'Processing...'}"
            print(output, end="", flush=True)
    
    async def _show_percentage(
        self,
        progress: float,
        total: Optional[float],
        message: Optional[str]
    ) -> None:
        """Show percentage only."""
        if total:
            percentage = (progress / total) * 100
            output = f"\r{percentage:.1f}%"
            if message:
                output += f" - {message}"
            print(output, end="", flush=True)
            
            if progress >= total:
                print()
        else:
            print(f"\rProgress: {progress:.1f} - {message or ''}", end="", flush=True)
    
    async def _show_dots(
        self,
        progress: float,
        total: Optional[float],
        message: Optional[str]
    ) -> None:
        """Show progress dots."""
        # Only update every 10% or every second
        now = datetime.now()
        
        if total:
            percentage = int((progress / total) * 10)
            key = f"{total}-{percentage}"
        else:
            # Update every second for indeterminate
            key = int(now.timestamp())
        
        if key != self.last_progress.get("key"):
            self.last_progress["key"] = key
            print(".", end="", flush=True)
            
            if total and progress >= total:
                print(f" Complete! {message or ''}")


class SamplingHandler:
    """
    Handle sampling requests (LLM calls) from MCP server.
    Integrates with various LLM providers.
    """
    
    def __init__(
        self,
        provider: str = "mock",
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize sampling handler.
        
        Args:
            provider: "gemini", "openai", "anthropic", or "mock"
            api_key: API key for provider
            model: Default model to use
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        self.default_model = model
        
        # Initialize provider
        if provider == "gemini" and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
        elif provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
    
    async def handle(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle sampling request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            params: Model parameters (model, temperature, max_tokens, etc.)
            context: Additional context
        
        Returns:
            Dict with 'content' and optional 'model', 'usage' keys
        """
        if self.provider == "gemini":
            return await self._gemini_sample(messages, params, context)
        elif self.provider == "openai":
            return await self._openai_sample(messages, params, context)
        elif self.provider == "anthropic":
            return await self._anthropic_sample(messages, params, context)
        else:  # mock
            return await self._mock_sample(messages, params, context)
    
    async def _gemini_sample(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample using Google Gemini."""
        if not GEMINI_AVAILABLE:
            return await self._mock_sample(messages, params, context)
        
        try:
            model = genai.GenerativeModel(
                params.get("model", self.default_model or "gemini-2.0-flash")
            )
            
            # Convert messages to Gemini format
            prompt = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            ])
            
            # Generate
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": params.get("temperature", 0.7),
                    "max_output_tokens": params.get("max_tokens", 500),
                }
            )
            
            return {
                "content": response.text,
                "model": model.model_name,
                "usage": {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            }
            
        except Exception as e:
            logger.error(f"Gemini sampling error: {e}")
            return {"content": f"Error: {e}", "error": True}
    
    async def _openai_sample(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample using OpenAI."""
        if not OPENAI_AVAILABLE:
            return await self._mock_sample(messages, params, context)
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=params.get("model", self.default_model or "gpt-3.5-turbo"),
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 500),
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI sampling error: {e}")
            return {"content": f"Error: {e}", "error": True}
    
    async def _anthropic_sample(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sample using Anthropic Claude."""
        # Would require anthropic SDK
        return await self._mock_sample(messages, params, context)
    
    async def _mock_sample(
        self,
        messages: List[Dict[str, str]],
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock sampling for testing."""
        # Get last user message
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate mock response based on content
        if "code" in user_message.lower():
            content = """```python
def example():
    return "Generated code"
```"""
        elif "analyze" in user_message.lower():
            content = "Analysis: The input appears to be a request for analysis."
        elif "summary" in user_message.lower():
            content = "Summary: This is a mock summary of the provided content."
        else:
            content = f"Mock response for: {user_message[:50]}..."
        
        return {
            "content": content,
            "model": "mock-model",
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(content.split()),
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + len(content.split())
            }
        }


# Example usage
async def example_usage():
    """Example of using handlers with FastMCP client."""
    from fastmcp import Client
    
    # Create handlers
    elicitation = ElicitationHandler(mode="interactive")
    progress = ProgressHandler(mode="bar")
    sampling = SamplingHandler(provider="mock")
    
    # Create client with handlers
    async with Client(
        "server_advanced.py",
        elicitation_handler=elicitation.handle,
        progress_handler=progress.handle,
        sampling_handler=sampling.handle
    ) as client:
        # Test tools that use handlers
        
        # Tool that uses elicitation
        result = await client.call_tool("interactive_setup", {})
        print(f"Setup result: {result}")
        
        # Tool that reports progress
        result = await client.call_tool(
            "process_batch",
            {"items": ["item1", "item2", "item3"]}
        )
        print(f"Batch result: {result}")
        
        # Tool that uses sampling
        result = await client.call_tool(
            "generate_code",
            {
                "language": "python",
                "description": "fibonacci function",
                "style": "recursive"
            }
        )
        print(f"Generated code: {result}")


if __name__ == "__main__":
    # Test handlers
    asyncio.run(example_usage())