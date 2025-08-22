"""
Dynamic Resources
=================
Dynamic resources with template support.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import random

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Config, get_cache_value, set_cache_value

logger = logging.getLogger(__name__)


async def get_user_profile(user_id: str) -> str:
    """
    Get user profile by ID.
    
    Args:
        user_id: User identifier
    
    Returns:
        User profile as formatted text
    """
    try:
        # Check cache first
        cache_key = f"user_profile_{user_id}"
        cached = await get_cache_value(cache_key)
        
        if cached:
            logger.debug(f"Cache hit for user {user_id}")
            return cached
        
        # Simulate fetching user data
        # In real implementation, would fetch from database or API
        profile_data = {
            "id": user_id,
            "username": f"user_{user_id}",
            "email": f"user_{user_id}@example.com",
            "created": "2024-01-01T00:00:00Z",
            "last_active": datetime.now().isoformat(),
            "status": "active",
            "preferences": {
                "theme": "dark",
                "notifications": True,
                "language": "en"
            },
            "stats": {
                "projects": random.randint(1, 10),
                "tasks_completed": random.randint(10, 100),
                "api_calls": random.randint(100, 1000)
            }
        }
        
        # Format as text
        profile = f"""User Profile: {profile_data['username']}
========================================

ID: {profile_data['id']}
Email: {profile_data['email']}
Status: {profile_data['status']}
Created: {profile_data['created']}
Last Active: {profile_data['last_active']}

Preferences:
- Theme: {profile_data['preferences']['theme']}
- Notifications: {profile_data['preferences']['notifications']}
- Language: {profile_data['preferences']['language']}

Statistics:
- Projects: {profile_data['stats']['projects']}
- Tasks Completed: {profile_data['stats']['tasks_completed']}
- API Calls: {profile_data['stats']['api_calls']}

Generated: {datetime.now().isoformat()}
"""
        
        # Cache the result
        await set_cache_value(cache_key, profile, ttl=300)
        
        return profile
        
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        return f"Error loading user profile: {str(e)}"


async def get_project_data(project_id: str) -> str:
    """
    Get project data by ID.
    
    Args:
        project_id: Project identifier
    
    Returns:
        Project data as formatted JSON
    """
    try:
        # Check cache
        cache_key = f"project_{project_id}"
        cached = await get_cache_value(cache_key)
        
        if cached:
            logger.debug(f"Cache hit for project {project_id}")
            return cached
        
        # Simulate project data
        project = {
            "id": project_id,
            "name": f"Project {project_id}",
            "description": f"Description for project {project_id}",
            "status": random.choice(["active", "pending", "completed"]),
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": datetime.now().isoformat(),
            "owner": f"user_{random.randint(1, 100)}",
            "team_members": [
                f"user_{random.randint(1, 100)}" 
                for _ in range(random.randint(1, 5))
            ],
            "metadata": {
                "priority": random.choice(["low", "medium", "high"]),
                "tags": ["tag1", "tag2", "tag3"][:random.randint(1, 3)],
                "category": random.choice(["development", "research", "operations"])
            },
            "statistics": {
                "tasks_total": random.randint(10, 100),
                "tasks_completed": random.randint(5, 50),
                "progress_percentage": random.randint(0, 100),
                "estimated_completion": (
                    datetime.now() + timedelta(days=random.randint(1, 90))
                ).isoformat()
            },
            "resources": {
                "documents": random.randint(0, 20),
                "images": random.randint(0, 50),
                "datasets": random.randint(0, 10)
            }
        }
        
        # Format as JSON
        result = json.dumps(project, indent=2)
        
        # Cache the result
        await set_cache_value(cache_key, result, ttl=300)
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting project data: {e}")
        return json.dumps({"error": str(e)})


async def get_api_endpoint(version: str, endpoint: str) -> str:
    """
    Get API endpoint documentation.
    
    Args:
        version: API version (v1, v2, etc.)
        endpoint: Endpoint path
    
    Returns:
        API endpoint documentation
    """
    try:
        # Define available endpoints
        endpoints = {
            "v1": {
                "users": {
                    "methods": ["GET", "POST"],
                    "description": "User management",
                    "parameters": ["id", "username", "email"]
                },
                "projects": {
                    "methods": ["GET", "POST", "PUT", "DELETE"],
                    "description": "Project management",
                    "parameters": ["id", "name", "status"]
                },
                "tasks": {
                    "methods": ["GET", "POST", "PUT", "DELETE"],
                    "description": "Task management",
                    "parameters": ["id", "project_id", "assignee", "status"]
                }
            },
            "v2": {
                "users": {
                    "methods": ["GET", "POST", "PATCH", "DELETE"],
                    "description": "Enhanced user management",
                    "parameters": ["id", "username", "email", "role", "permissions"]
                },
                "projects": {
                    "methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": "Advanced project management",
                    "parameters": ["id", "name", "status", "team", "metadata"]
                },
                "analytics": {
                    "methods": ["GET"],
                    "description": "Analytics and reporting",
                    "parameters": ["metric", "start_date", "end_date", "granularity"]
                }
            }
        }
        
        # Get endpoint info
        if version in endpoints and endpoint in endpoints[version]:
            info = endpoints[version][endpoint]
            
            doc = f"""API Endpoint: /{version}/{endpoint}
=====================================

Description: {info['description']}

Supported Methods: {', '.join(info['methods'])}

Parameters:
{chr(10).join(f'  - {param}' for param in info['parameters'])}

Example Request:
```
{info['methods'][0]} /api/{version}/{endpoint}
Content-Type: application/json
Authorization: Bearer <token>
```

Example Response:
```json
{{
  "success": true,
  "data": {{
    // Response data here
  }},
  "timestamp": "{datetime.now().isoformat()}"
}}
```

Rate Limits:
- 100 requests per minute
- 1000 requests per hour

Authentication:
- API Key required
- Bearer token supported
- OAuth2 available for v2

Generated: {datetime.now().isoformat()}
"""
            return doc
        else:
            return f"Endpoint not found: /{version}/{endpoint}"
            
    except Exception as e:
        logger.error(f"Error getting API endpoint: {e}")
        return f"Error: {str(e)}"


async def get_statistics(
    entity_type: Optional[str] = None,
    time_range: Optional[str] = "24h"
) -> str:
    """
    Get system statistics.
    
    Args:
        entity_type: Type of entity to get stats for
        time_range: Time range for statistics
    
    Returns:
        Statistics as formatted text
    """
    try:
        # Generate sample statistics
        stats = {
            "time_range": time_range,
            "generated_at": datetime.now().isoformat(),
            "system": {
                "uptime_hours": random.randint(1, 720),
                "total_requests": random.randint(1000, 100000),
                "active_users": random.randint(10, 1000),
                "error_rate": round(random.random() * 5, 2)
            },
            "performance": {
                "avg_response_time_ms": random.randint(50, 500),
                "p95_response_time_ms": random.randint(100, 1000),
                "p99_response_time_ms": random.randint(200, 2000),
                "cache_hit_rate": round(random.random() * 100, 2)
            }
        }
        
        # Add entity-specific stats
        if entity_type:
            if entity_type == "users":
                stats["users"] = {
                    "total": random.randint(100, 10000),
                    "active_today": random.randint(10, 1000),
                    "new_this_week": random.randint(1, 100)
                }
            elif entity_type == "projects":
                stats["projects"] = {
                    "total": random.randint(10, 1000),
                    "active": random.randint(5, 500),
                    "completed_this_month": random.randint(1, 50)
                }
            elif entity_type == "api":
                stats["api"] = {
                    "total_calls": random.randint(10000, 1000000),
                    "unique_endpoints": random.randint(10, 100),
                    "avg_payload_size_kb": random.randint(1, 100)
                }
        
        # Format as text
        result = f"""System Statistics
=================
Time Range: {stats['time_range']}
Generated: {stats['generated_at']}

System Metrics:
- Uptime: {stats['system']['uptime_hours']} hours
- Total Requests: {stats['system']['total_requests']:,}
- Active Users: {stats['system']['active_users']:,}
- Error Rate: {stats['system']['error_rate']}%

Performance Metrics:
- Avg Response Time: {stats['performance']['avg_response_time_ms']}ms
- P95 Response Time: {stats['performance']['p95_response_time_ms']}ms
- P99 Response Time: {stats['performance']['p99_response_time_ms']}ms
- Cache Hit Rate: {stats['performance']['cache_hit_rate']}%
"""
        
        # Add entity-specific section
        if entity_type in stats:
            result += f"\n{entity_type.capitalize()} Statistics:\n"
            for key, value in stats[entity_type].items():
                formatted_key = key.replace('_', ' ').title()
                formatted_value = f"{value:,}" if isinstance(value, int) else value
                result += f"- {formatted_key}: {formatted_value}\n"
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return f"Error: {str(e)}"


async def get_analytics_report(
    report_type: str = "summary",
    format: str = "text"
) -> str:
    """
    Get analytics report.
    
    Args:
        report_type: Type of report (summary, detailed, custom)
        format: Output format (text, json)
    
    Returns:
        Analytics report in requested format
    """
    try:
        # Generate sample report data
        report_data = {
            "type": report_type,
            "generated": datetime.now().isoformat(),
            "period": {
                "start": (datetime.now() - timedelta(days=30)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "summary": {
                "total_revenue": random.randint(10000, 1000000),
                "total_users": random.randint(100, 10000),
                "conversion_rate": round(random.random() * 100, 2),
                "growth_rate": round(random.random() * 50 - 10, 2)
            },
            "trends": {
                "user_growth": "increasing",
                "engagement": "stable",
                "performance": "improving"
            },
            "top_metrics": [
                {"name": "Daily Active Users", "value": random.randint(100, 5000)},
                {"name": "Session Duration", "value": f"{random.randint(1, 30)} min"},
                {"name": "Page Views", "value": random.randint(1000, 100000)},
                {"name": "Bounce Rate", "value": f"{random.randint(20, 60)}%"}
            ]
        }
        
        if format == "json":
            return json.dumps(report_data, indent=2)
        else:
            # Format as text report
            report = f"""Analytics Report: {report_type.capitalize()}
{'=' * 50}

Report Period: {report_data['period']['start']} to {report_data['period']['end']}
Generated: {report_data['generated']}

EXECUTIVE SUMMARY
-----------------
Total Revenue: ${report_data['summary']['total_revenue']:,}
Total Users: {report_data['summary']['total_users']:,}
Conversion Rate: {report_data['summary']['conversion_rate']}%
Growth Rate: {report_data['summary']['growth_rate']:+.2f}%

TRENDS
------
• User Growth: {report_data['trends']['user_growth']}
• Engagement: {report_data['trends']['engagement']}
• Performance: {report_data['trends']['performance']}

TOP METRICS
-----------"""
            
            for metric in report_data['top_metrics']:
                report += f"\n• {metric['name']}: {metric['value']}"
            
            if report_type == "detailed":
                report += """

DETAILED ANALYSIS
-----------------
User Acquisition:
  - Organic: 45%
  - Paid: 30%
  - Referral: 15%
  - Direct: 10%

Geographic Distribution:
  - North America: 40%
  - Europe: 30%
  - Asia Pacific: 20%
  - Other: 10%

Device Breakdown:
  - Desktop: 55%
  - Mobile: 35%
  - Tablet: 10%

RECOMMENDATIONS
---------------
1. Focus on mobile optimization to capture growing mobile traffic
2. Increase investment in high-performing acquisition channels
3. Implement A/B testing for conversion rate optimization
4. Enhance user engagement features based on usage patterns"""
            
            return report
            
    except Exception as e:
        logger.error(f"Error generating analytics report: {e}")
        return f"Error: {str(e)}"