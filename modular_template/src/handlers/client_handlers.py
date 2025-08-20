"""
Client Handlers
===============
Handle client-specific events and messages.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


async def handle_client_message(
    message: Dict[str, Any],
    client_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle incoming client messages.
    
    Args:
        message: Client message
        client_id: Optional client identifier
    
    Returns:
        Response to client
    """
    try:
        message_type = message.get("type", "unknown")
        logger.debug(f"Received {message_type} message from client {client_id}")
        
        # Process different message types
        if message_type == "ping":
            return {
                "type": "pong",
                "timestamp": datetime.now().isoformat(),
                "client_id": client_id
            }
        
        elif message_type == "status":
            return {
                "type": "status_response",
                "status": "active",
                "timestamp": datetime.now().isoformat()
            }
        
        elif message_type == "config":
            # Handle configuration requests
            config_key = message.get("key")
            if config_key:
                logger.info(f"Client {client_id} requested config: {config_key}")
                # Return config value (implement as needed)
                return {
                    "type": "config_response",
                    "key": config_key,
                    "value": None,  # Get from config
                    "timestamp": datetime.now().isoformat()
                }
        
        elif message_type == "metric":
            # Handle metric submissions
            metric_name = message.get("name")
            metric_value = message.get("value")
            if metric_name:
                logger.info(f"Metric from {client_id}: {metric_name}={metric_value}")
                return {
                    "type": "metric_ack",
                    "name": metric_name,
                    "timestamp": datetime.now().isoformat()
                }
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return {
                "type": "error",
                "error": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error handling client message: {e}")
        return {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_client_error(
    error: Exception,
    client_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handle client errors.
    
    Args:
        error: The error that occurred
        client_id: Optional client identifier
        context: Optional error context
    
    Returns:
        Error response
    """
    try:
        error_type = type(error).__name__
        error_message = str(error)
        
        logger.error(f"Client error for {client_id}: {error_type} - {error_message}")
        
        # Log additional context if provided
        if context:
            logger.error(f"Error context: {context}")
        
        # Determine error severity
        if isinstance(error, (ConnectionError, TimeoutError)):
            severity = "network"
        elif isinstance(error, (ValueError, TypeError)):
            severity = "validation"
        elif isinstance(error, PermissionError):
            severity = "permission"
        else:
            severity = "unknown"
        
        response = {
            "type": "error_handled",
            "error_type": error_type,
            "error_message": error_message,
            "severity": severity,
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add recovery suggestions based on error type
        if severity == "network":
            response["recovery"] = "Check network connection and retry"
        elif severity == "validation":
            response["recovery"] = "Verify input data format"
        elif severity == "permission":
            response["recovery"] = "Check access permissions"
        
        return response
        
    except Exception as e:
        logger.critical(f"Failed to handle client error: {e}")
        return {
            "type": "critical_error",
            "error": "Error handler failed",
            "original_error": str(error),
            "handler_error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_client_disconnect(
    client_id: str,
    reason: Optional[str] = None,
    cleanup: bool = True
) -> Dict[str, Any]:
    """
    Handle client disconnection.
    
    Args:
        client_id: Client identifier
        reason: Disconnection reason
        cleanup: Whether to perform cleanup
    
    Returns:
        Disconnection acknowledgment
    """
    try:
        logger.info(f"Client {client_id} disconnecting: {reason or 'No reason provided'}")
        
        if cleanup:
            # Perform cleanup tasks
            logger.debug(f"Cleaning up resources for client {client_id}")
            
            # Clear any client-specific cache
            # await clear_client_cache(client_id)
            
            # Cancel any pending operations
            # await cancel_client_operations(client_id)
            
            # Update client status
            # await update_client_status(client_id, "disconnected")
        
        return {
            "type": "disconnect_ack",
            "client_id": client_id,
            "reason": reason,
            "cleanup_performed": cleanup,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error during client disconnect: {e}")
        return {
            "type": "disconnect_error",
            "client_id": client_id,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Client tracking utilities
_active_clients: Dict[str, Dict[str, Any]] = {}


async def register_client(
    client_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Register a new client.
    
    Args:
        client_id: Client identifier
        metadata: Optional client metadata
    
    Returns:
        True if registered successfully
    """
    try:
        _active_clients[client_id] = {
            "connected_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "last_activity": datetime.now().isoformat()
        }
        logger.info(f"Client {client_id} registered")
        return True
    except Exception as e:
        logger.error(f"Failed to register client {client_id}: {e}")
        return False


async def unregister_client(client_id: str) -> bool:
    """
    Unregister a client.
    
    Args:
        client_id: Client identifier
    
    Returns:
        True if unregistered successfully
    """
    try:
        if client_id in _active_clients:
            del _active_clients[client_id]
            logger.info(f"Client {client_id} unregistered")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to unregister client {client_id}: {e}")
        return False


async def update_client_activity(client_id: str) -> bool:
    """
    Update client's last activity timestamp.
    
    Args:
        client_id: Client identifier
    
    Returns:
        True if updated successfully
    """
    try:
        if client_id in _active_clients:
            _active_clients[client_id]["last_activity"] = datetime.now().isoformat()
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to update client activity: {e}")
        return False


async def get_client_info(client_id: str) -> Optional[Dict[str, Any]]:
    """
    Get client information.
    
    Args:
        client_id: Client identifier
    
    Returns:
        Client information or None
    """
    return _active_clients.get(client_id)


async def list_active_clients() -> list[str]:
    """
    List all active client IDs.
    
    Returns:
        List of active client IDs
    """
    return list(_active_clients.keys())