"""
Resources Module
================
Static and dynamic resources for the MCP server.
"""

from .static import (
    server_info,
    configuration,
    documentation
)

from .dynamic import (
    get_user_profile,
    get_project_data,
    get_api_endpoint,
    get_statistics,
    get_analytics_report
)

__all__ = [
    # Static resources
    'server_info',
    'configuration',
    'documentation',
    
    # Dynamic resources (templates)
    'get_user_profile',
    'get_project_data', 
    'get_api_endpoint',
    'get_statistics',
    'get_analytics_report'
]