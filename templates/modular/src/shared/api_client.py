"""
API Client Module
=================
HTTP client for external API integrations.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager
import aiohttp
from aiohttp import ClientSession, ClientTimeout, ClientError

from .config import Config
from .utils import format_error

logger = logging.getLogger(__name__)


class APIClient:
    """Async HTTP client with connection pooling and retry logic."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL for API requests
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.base_url = base_url or Config.API_BASE_URL
        self.api_key = api_key or Config.API_KEY
        self.timeout = timeout or Config.REQUEST_TIMEOUT
        self.max_retries = max_retries or Config.MAX_RETRIES
        self.session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the client session."""
        if self.session is None or self.session.closed:
            async with self._lock:
                if self.session is None or self.session.closed:
                    timeout = ClientTimeout(total=self.timeout)
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        ttl_dns_cache=300
                    )
                    
                    headers = {}
                    if self.api_key:
                        headers["Authorization"] = f"Bearer {self.api_key}"
                    
                    self.session = ClientSession(
                        base_url=self.base_url,
                        timeout=timeout,
                        connector=connector,
                        headers=headers
                    )
                    logger.info(f"API client session started for {self.base_url}")
    
    async def close(self):
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("API client session closed")
    
    async def request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request parameters
        
        Returns:
            Response data
        """
        if not self.session or self.session.closed:
            await self.start()
        
        url = endpoint if endpoint.startswith('http') else f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    response.raise_for_status()
                    
                    # Parse response based on content type
                    content_type = response.headers.get('Content-Type', '')
                    
                    if 'application/json' in content_type:
                        data = await response.json()
                    elif 'text' in content_type:
                        data = await response.text()
                    else:
                        data = await response.read()
                    
                    logger.debug(f"Successful {method} request to {url}")
                    return {
                        "status": response.status,
                        "data": data,
                        "headers": dict(response.headers)
                    }
                    
            except ClientError as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        raise Exception(f"Failed after {self.max_retries} attempts")
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a GET request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional parameters
        
        Returns:
            Response data
        """
        return await self.request("GET", endpoint, **kwargs)
    
    async def post(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a POST request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional parameters
        
        Returns:
            Response data
        """
        return await self.request("POST", endpoint, **kwargs)
    
    async def put(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a PUT request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional parameters
        
        Returns:
            Response data
        """
        return await self.request("PUT", endpoint, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a DELETE request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional parameters
        
        Returns:
            Response data
        """
        return await self.request("DELETE", endpoint, **kwargs)
    
    async def patch(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make a PATCH request.
        
        Args:
            endpoint: API endpoint
            **kwargs: Additional parameters
        
        Returns:
            Response data
        """
        return await self.request("PATCH", endpoint, **kwargs)
    
    @asynccontextmanager
    async def stream_response(self, method: str, endpoint: str, **kwargs):
        """
        Stream response data.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional parameters
        
        Yields:
            Response stream
        """
        if not self.session or self.session.closed:
            await self.start()
        
        url = endpoint if endpoint.startswith('http') else f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            yield response
    
    async def download_file(
        self,
        endpoint: str,
        file_path: str,
        chunk_size: int = 8192
    ) -> Dict[str, Any]:
        """
        Download a file from an endpoint.
        
        Args:
            endpoint: API endpoint
            file_path: Local file path to save to
            chunk_size: Download chunk size
        
        Returns:
            Download statistics
        """
        try:
            async with self.stream_response("GET", endpoint) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(file_path, 'wb') as file:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        file.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
                
                logger.info(f"Downloaded {downloaded} bytes to {file_path}")
                
                return {
                    "file_path": file_path,
                    "size": downloaded,
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise
    
    async def upload_file(
        self,
        endpoint: str,
        file_path: str,
        field_name: str = "file",
        additional_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Upload a file to an endpoint.
        
        Args:
            endpoint: API endpoint
            file_path: Local file path to upload
            field_name: Form field name for the file
            additional_data: Additional form data
        
        Returns:
            Upload response
        """
        try:
            data = aiohttp.FormData()
            
            # Add file
            with open(file_path, 'rb') as file:
                data.add_field(
                    field_name,
                    file,
                    filename=file_path.split('/')[-1]
                )
            
            # Add additional fields
            if additional_data:
                for key, value in additional_data.items():
                    data.add_field(key, str(value))
            
            response = await self.post(endpoint, data=data)
            logger.info(f"Uploaded {file_path} to {endpoint}")
            
            return response
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise


# Global client instance
_api_client: Optional[APIClient] = None


async def get_api_client() -> APIClient:
    """
    Get the global API client instance.
    
    Returns:
        API client instance
    """
    global _api_client
    
    if _api_client is None:
        _api_client = APIClient()
        await _api_client.start()
    
    return _api_client


async def verify_api_connection() -> bool:
    """
    Verify API connection is working.
    
    Returns:
        True if connection is successful
    """
    try:
        if not Config.API_BASE_URL:
            logger.info("No API base URL configured")
            return True
        
        client = await get_api_client()
        
        # Try a simple health check or root endpoint
        try:
            await client.get("/health")
        except:
            # Try root if health doesn't exist
            await client.get("/")
        
        logger.info("API connection verified")
        return True
        
    except Exception as e:
        logger.error(f"API connection failed: {e}")
        return False


async def close_api_client():
    """Close the global API client."""
    global _api_client
    
    if _api_client:
        await _api_client.close()
        _api_client = None
        logger.info("API client closed")


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_second: float = 10):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.rate = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait if necessary to respect rate limit."""
        async with self._lock:
            current = asyncio.get_event_loop().time()
            time_since_last = current - self.last_request
            
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self.last_request = asyncio.get_event_loop().time()


class ConnectionPool:
    """Connection pool for managing multiple API clients."""
    
    def __init__(self, size: int = 5):
        """
        Initialize connection pool.
        
        Args:
            size: Pool size
        """
        self.size = size
        self.clients: list[APIClient] = []
        self.available: asyncio.Queue = asyncio.Queue(maxsize=size)
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            for _ in range(self.size):
                client = APIClient()
                await client.start()
                self.clients.append(client)
                await self.available.put(client)
            
            self._initialized = True
            logger.info(f"Connection pool initialized with {self.size} clients")
    
    @asynccontextmanager
    async def get_client(self):
        """
        Get a client from the pool.
        
        Yields:
            API client
        """
        if not self._initialized:
            await self.initialize()
        
        client = await self.available.get()
        try:
            yield client
        finally:
            await self.available.put(client)
    
    async def close(self):
        """Close all clients in the pool."""
        for client in self.clients:
            await client.close()
        
        self.clients.clear()
        self._initialized = False
        logger.info("Connection pool closed")


# Optional: Global connection pool
_connection_pool: Optional[ConnectionPool] = None


async def get_connection_pool(size: int = 5) -> ConnectionPool:
    """
    Get the global connection pool.
    
    Args:
        size: Pool size
    
    Returns:
        Connection pool instance
    """
    global _connection_pool
    
    if _connection_pool is None:
        _connection_pool = ConnectionPool(size)
        await _connection_pool.initialize()
    
    return _connection_pool