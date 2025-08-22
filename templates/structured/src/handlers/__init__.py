"""
Handlers Module
===============
Event and client handlers for the MCP server.
"""

from .client_handlers import (
    handle_client_message,
    handle_client_error,
    handle_client_disconnect
)
from .event_handlers import (
    handle_startup,
    handle_shutdown,
    handle_request,
    handle_response
)

__all__ = [
    # Client handlers
    'handle_client_message',
    'handle_client_error', 
    'handle_client_disconnect',
    
    # Event handlers
    'handle_startup',
    'handle_shutdown',
    'handle_request',
    'handle_response'
]