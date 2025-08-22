"""
API Integration Tools
=====================
Tools for interacting with external APIs.
"""

import logging
import json
from typing import Dict, Any, Optional, List
import aiohttp
from datetime import datetime

from ..utils import format_success, format_error, get_api_client

logger = logging.getLogger(__name__)


async def fetch_api_data(
    endpoint: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Fetch data from an API endpoint.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        params: Query parameters
        headers: Additional headers
    
    Returns:
        API response data
    """
    try:
        client = await get_api_client()
        
        # Build request
        request_kwargs = {
            "method": method,
            "url": endpoint,
            "params": params or {},
            "headers": headers or {}
        }
        
        # Make request
        response = await client.request(**request_kwargs)
        
        # Check status
        if response.status >= 400:
            return format_error(f"API error: {response.status} - {response.reason}")
        
        # Parse response
        data = await response.json()
        
        return format_success({
            "endpoint": endpoint,
            "method": method,
            "status": response.status,
            "data": data
        })
        
    except aiohttp.ClientError as e:
        logger.error(f"API client error: {e}")
        return format_error(f"Connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Error fetching API data: {e}")
        return format_error(str(e))


async def post_api_data(
    endpoint: str,
    data: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    content_type: str = "application/json"
) -> Dict[str, Any]:
    """
    Post data to an API endpoint.
    
    Args:
        endpoint: API endpoint path
        data: Data to post
        headers: Additional headers
        content_type: Content type for the request
    
    Returns:
        API response
    """
    try:
        client = await get_api_client()
        
        # Prepare headers
        request_headers = headers or {}
        request_headers["Content-Type"] = content_type
        
        # Prepare data based on content type
        if content_type == "application/json":
            request_data = json.dumps(data)
        else:
            request_data = data
        
        # Make request
        response = await client.post(
            endpoint,
            data=request_data,
            headers=request_headers
        )
        
        # Check status
        if response.status >= 400:
            error_text = await response.text()
            return format_error(f"API error: {response.status} - {error_text}")
        
        # Parse response
        if response.headers.get("Content-Type", "").startswith("application/json"):
            result = await response.json()
        else:
            result = await response.text()
        
        return format_success({
            "endpoint": endpoint,
            "status": response.status,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error posting API data: {e}")
        return format_error(str(e))


async def update_api_resource(
    endpoint: str,
    resource_id: str,
    updates: Dict[str, Any],
    method: str = "PUT"
) -> Dict[str, Any]:
    """
    Update an API resource.
    
    Args:
        endpoint: Base endpoint path
        resource_id: Resource identifier
        updates: Update data
        method: HTTP method (PUT or PATCH)
    
    Returns:
        Update result
    """
    try:
        client = await get_api_client()
        
        # Build URL
        url = f"{endpoint}/{resource_id}"
        
        # Make request
        request_data = json.dumps(updates)
        headers = {"Content-Type": "application/json"}
        
        if method == "PUT":
            response = await client.put(url, data=request_data, headers=headers)
        elif method == "PATCH":
            response = await client.patch(url, data=request_data, headers=headers)
        else:
            return format_error(f"Invalid method: {method}")
        
        # Check status
        if response.status >= 400:
            error_text = await response.text()
            return format_error(f"Update failed: {response.status} - {error_text}")
        
        # Parse response
        if response.headers.get("Content-Type", "").startswith("application/json"):
            result = await response.json()
        else:
            result = {"status": "updated", "resource_id": resource_id}
        
        return format_success({
            "resource_id": resource_id,
            "updates_applied": list(updates.keys()),
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error updating API resource: {e}")
        return format_error(str(e))


async def delete_api_resource(
    endpoint: str,
    resource_id: str,
    confirm: bool = False
) -> Dict[str, Any]:
    """
    Delete an API resource.
    
    Args:
        endpoint: Base endpoint path
        resource_id: Resource identifier
        confirm: Confirmation flag for safety
    
    Returns:
        Deletion result
    """
    try:
        if not confirm:
            return format_error("Deletion requires confirmation (set confirm=True)")
        
        client = await get_api_client()
        
        # Build URL
        url = f"{endpoint}/{resource_id}"
        
        # Make request
        response = await client.delete(url)
        
        # Check status
        if response.status >= 400:
            error_text = await response.text()
            return format_error(f"Deletion failed: {response.status} - {error_text}")
        
        return format_success({
            "resource_id": resource_id,
            "deleted": True,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error deleting API resource: {e}")
        return format_error(str(e))