"""
Utility Tools
=============
General utility and diagnostic tools.
"""

import logging
import sys
import os
import platform
import psutil
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import format_success, format_error, Config

logger = logging.getLogger(__name__)


async def health_check(
    include_dependencies: bool = True,
    include_resources: bool = True
) -> Dict[str, Any]:
    """
    Perform server health check.
    
    Args:
        include_dependencies: Check external dependencies
        include_resources: Check system resources
    
    Returns:
        Health status and diagnostics
    """
    try:
        health_status = {
            "healthy": True,
            "timestamp": datetime.now().isoformat(),
            "server": {
                "name": Config.SERVER_NAME,
                "version": Config.SERVER_VERSION,
                "environment": Config.ENVIRONMENT,
                "uptime": None  # Would need to track server start time
            },
            "checks": []
        }
        
        # Check system resources
        if include_resources:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                resource_check = {
                    "name": "system_resources",
                    "status": "healthy",
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available_gb": memory.available / (1024**3),
                        "disk_percent": disk.percent,
                        "disk_free_gb": disk.free / (1024**3)
                    }
                }
                
                # Flag unhealthy conditions
                if cpu_percent > 90:
                    resource_check["status"] = "warning"
                    resource_check["message"] = "High CPU usage"
                if memory.percent > 90:
                    resource_check["status"] = "critical"
                    resource_check["message"] = "High memory usage"
                    health_status["healthy"] = False
                if disk.percent > 95:
                    resource_check["status"] = "critical"
                    resource_check["message"] = "Low disk space"
                    health_status["healthy"] = False
                
                health_status["checks"].append(resource_check)
                
            except Exception as e:
                health_status["checks"].append({
                    "name": "system_resources",
                    "status": "error",
                    "message": str(e)
                })
        
        # Check dependencies
        if include_dependencies:
            # Check API connection if configured
            if Config.API_BASE_URL:
                try:
                    from shared import verify_api_connection
                    api_healthy = await verify_api_connection()
                    
                    health_status["checks"].append({
                        "name": "api_connection",
                        "status": "healthy" if api_healthy else "unhealthy",
                        "url": Config.API_BASE_URL
                    })
                    
                    if not api_healthy:
                        health_status["healthy"] = False
                        
                except Exception as e:
                    health_status["checks"].append({
                        "name": "api_connection",
                        "status": "error",
                        "message": str(e)
                    })
                    health_status["healthy"] = False
            
            # Check cache if enabled
            if Config.ENABLE_CACHE:
                try:
                    from shared import get_cache_stats
                    cache_stats = await get_cache_stats()
                    
                    health_status["checks"].append({
                        "name": "cache",
                        "status": "healthy",
                        "details": cache_stats
                    })
                except Exception as e:
                    health_status["checks"].append({
                        "name": "cache",
                        "status": "error",
                        "message": str(e)
                    })
        
        return format_success(health_status)
        
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return format_error(str(e))


async def get_status(
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Get current server status.
    
    Args:
        verbose: Include detailed information
    
    Returns:
        Server status information
    """
    try:
        status = {
            "server": {
                "name": Config.SERVER_NAME,
                "version": Config.SERVER_VERSION,
                "environment": Config.ENVIRONMENT,
                "transport": Config.TRANSPORT
            },
            "system": {
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
                "hostname": platform.node()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        if verbose:
            # Add detailed system info
            status["system"].update({
                "processor": platform.processor(),
                "architecture": platform.architecture()[0],
                "cpu_count": os.cpu_count()
            })
            
            # Add process info
            process = psutil.Process()
            status["process"] = {
                "pid": process.pid,
                "memory_mb": process.memory_info().rss / (1024 * 1024),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            }
            
            # Add configuration info
            status["configuration"] = {
                "log_level": Config.LOG_LEVEL,
                "cache_enabled": Config.ENABLE_CACHE,
                "cache_ttl": Config.CACHE_TTL,
                "max_retries": Config.MAX_RETRIES,
                "advanced_features": Config.ENABLE_ADVANCED_FEATURES
            }
        
        return format_success(status)
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return format_error(str(e))


async def run_diagnostics(
    test_api: bool = True,
    test_cache: bool = True,
    test_filesystem: bool = True,
    test_network: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive diagnostics.
    
    Args:
        test_api: Test API connectivity
        test_cache: Test cache operations
        test_filesystem: Test file system access
        test_network: Test network connectivity
    
    Returns:
        Diagnostic results
    """
    try:
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
        
        # Test API connectivity
        if test_api and Config.API_BASE_URL:
            test_result = {
                "name": "API Connectivity",
                "status": "pending"
            }
            
            try:
                from ..shared import verify_api_connection
                start = datetime.now()
                connected = await verify_api_connection()
                duration = (datetime.now() - start).total_seconds()
                
                test_result["status"] = "passed" if connected else "failed"
                test_result["duration_ms"] = duration * 1000
                test_result["details"] = {
                    "url": Config.API_BASE_URL,
                    "connected": connected
                }
                
            except Exception as e:
                test_result["status"] = "failed"
                test_result["error"] = str(e)
            
            diagnostics["tests"].append(test_result)
        
        # Test cache operations
        if test_cache and Config.ENABLE_CACHE:
            test_result = {
                "name": "Cache Operations",
                "status": "pending"
            }
            
            try:
                from ..shared.cache import cache_get, cache_set
                
                # Test write
                test_key = f"diagnostic_test_{datetime.now().timestamp()}"
                test_value = {"test": True, "timestamp": datetime.now().isoformat()}
                await cache_set(test_key, test_value, ttl=60)
                
                # Test read
                retrieved = await cache_get(test_key)
                
                if retrieved == test_value:
                    test_result["status"] = "passed"
                    test_result["details"] = {"operations": ["set", "get"]}
                else:
                    test_result["status"] = "failed"
                    test_result["error"] = "Cache value mismatch"
                    
            except Exception as e:
                test_result["status"] = "failed"
                test_result["error"] = str(e)
            
            diagnostics["tests"].append(test_result)
        
        # Test filesystem access
        if test_filesystem:
            test_result = {
                "name": "Filesystem Access",
                "status": "pending"
            }
            
            try:
                import tempfile
                from pathlib import Path
                
                # Test write and read
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    test_path = Path(f.name)
                    f.write("diagnostic test")
                
                # Test read
                content = test_path.read_text()
                
                # Cleanup
                test_path.unlink()
                
                test_result["status"] = "passed"
                test_result["details"] = {
                    "operations": ["create", "write", "read", "delete"],
                    "temp_dir": tempfile.gettempdir()
                }
                
            except Exception as e:
                test_result["status"] = "failed"
                test_result["error"] = str(e)
            
            diagnostics["tests"].append(test_result)
        
        # Test network connectivity
        if test_network:
            test_result = {
                "name": "Network Connectivity",
                "status": "pending"
            }
            
            try:
                import aiohttp
                
                test_urls = [
                    ("Google DNS", "https://8.8.8.8"),
                    ("Cloudflare DNS", "https://1.1.1.1")
                ]
                
                results = []
                async with aiohttp.ClientSession() as session:
                    for name, url in test_urls:
                        try:
                            start = datetime.now()
                            async with session.head(url, timeout=5) as response:
                                duration = (datetime.now() - start).total_seconds()
                                results.append({
                                    "target": name,
                                    "reachable": True,
                                    "response_ms": duration * 1000
                                })
                        except:
                            results.append({
                                "target": name,
                                "reachable": False
                            })
                
                test_result["status"] = "passed" if any(r["reachable"] for r in results) else "failed"
                test_result["details"] = results
                
            except Exception as e:
                test_result["status"] = "failed"
                test_result["error"] = str(e)
            
            diagnostics["tests"].append(test_result)
        
        # Update summary
        for test in diagnostics["tests"]:
            diagnostics["summary"]["total"] += 1
            if test["status"] == "passed":
                diagnostics["summary"]["passed"] += 1
            elif test["status"] == "failed":
                diagnostics["summary"]["failed"] += 1
            else:
                diagnostics["summary"]["skipped"] += 1
        
        # Overall health
        diagnostics["healthy"] = diagnostics["summary"]["failed"] == 0
        
        return format_success(diagnostics)
        
    except Exception as e:
        logger.error(f"Error running diagnostics: {e}")
        return format_error(str(e))