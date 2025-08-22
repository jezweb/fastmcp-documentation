# FastMCP Integration Guide

Comprehensive guide for integrating FastMCP with various APIs, LLMs, and frameworks.

## Table of Contents
1. [Gemini SDK Integration](#gemini-sdk-integration)
2. [OpenAI API Integration](#openai-api-integration)
3. [Anthropic Integration](#anthropic-integration)
4. [OpenAPI/Swagger Integration](#openapiswagger-integration)
5. [FastAPI Integration](#fastapi-integration)
6. [REST API Connection](#rest-api-connection)
7. [Authentication Methods](#authentication-methods)
8. [Permit.io Integration](#permitio-integration)
9. [WorkOS Authentication](#workos-authentication)

## Gemini SDK Integration

Google's Gemini SDK has experimental support for MCP servers, allowing direct tool calling.

### Server Setup
```python
from fastmcp import FastMCP
import random

mcp = FastMCP("gemini-server")

@mcp.tool()
def analyze_text(text: str, analysis_type: str = "sentiment") -> dict:
    """Analyze text with specified analysis type."""
    # Your analysis logic here
    return {
        "text": text,
        "type": analysis_type,
        "result": "positive"  # Example result
    }

@mcp.tool()
def generate_summary(text: str, max_length: int = 100) -> str:
    """Generate a summary of the provided text."""
    # Summary generation logic
    return text[:max_length] + "..."
```

### Gemini Client Integration
```python
import google.generativeai as genai
from fastmcp import Client
import asyncio

# Configure Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")

async def use_with_gemini():
    # Create MCP client
    async with Client("gemini-server.py") as client:
        # Get client session for Gemini
        session = client.get_session()
        
        # Create Gemini model with MCP tools
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            tools=session  # Pass MCP session as tools
        )
        
        # Gemini can now call MCP tools automatically
        response = model.generate_content(
            "Analyze this text for sentiment: 'I love using FastMCP!'"
        )
        print(response.text)

asyncio.run(use_with_gemini())
```

### Advanced Gemini Features
```python
# Server with complex tools for Gemini
@mcp.tool()
async def research_topic(
    topic: str,
    depth: str = "basic",  # basic, intermediate, advanced
    include_sources: bool = True
) -> dict:
    """Research a topic with configurable depth."""
    # Research implementation
    return {
        "topic": topic,
        "summary": "Research findings...",
        "sources": ["source1", "source2"] if include_sources else []
    }

# Gemini with system instructions
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    tools=session,
    system_instruction="You are a research assistant. Use the available tools to provide comprehensive answers."
)
```

## OpenAI API Integration

### Server Setup for OpenAI
```python
from fastmcp import FastMCP
import openai

mcp = FastMCP("openai-server")

# Store OpenAI client
openai_client = openai.AsyncOpenAI(api_key="YOUR_OPENAI_API_KEY")

@mcp.tool()
async def gpt_complete(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Get completion from OpenAI."""
    response = await openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@mcp.tool()
async def gpt_analyze_code(code: str, language: str = "python") -> dict:
    """Analyze code using GPT."""
    prompt = f"Analyze this {language} code for issues and improvements:\n{code}"
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a code review expert."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return {
        "analysis": response.choices[0].message.content,
        "language": language
    }
```

### Client Integration with OpenAI
```python
from fastmcp import Client
import openai

async def openai_with_mcp():
    async with Client("openai-server.py") as client:
        # List available tools
        tools = await client.list_tools()
        
        # Convert MCP tools to OpenAI function format
        openai_functions = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
            for tool in tools
        ]
        
        # Use with OpenAI function calling
        response = await openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Analyze this Python code: print('hello')"}],
            functions=openai_functions,
            function_call="auto"
        )
        
        # Handle function calls
        if response.choices[0].message.function_call:
            func_call = response.choices[0].message.function_call
            result = await client.call_tool(
                func_call.name,
                json.loads(func_call.arguments)
            )
            print(result)
```

## Anthropic Integration

### Server for Claude
```python
from fastmcp import FastMCP
import anthropic

mcp = FastMCP("claude-server")

claude = anthropic.AsyncAnthropic(api_key="YOUR_ANTHROPIC_API_KEY")

@mcp.tool()
async def claude_analyze(
    text: str,
    analysis_type: str = "general",
    max_tokens: int = 1000
) -> str:
    """Analyze text using Claude."""
    
    prompts = {
        "general": "Analyze the following text:",
        "sentiment": "Analyze the sentiment of:",
        "summary": "Summarize the following:",
        "critique": "Provide constructive criticism for:"
    }
    
    message = await claude.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=max_tokens,
        messages=[{
            "role": "user",
            "content": f"{prompts.get(analysis_type, prompts['general'])} {text}"
        }]
    )
    
    return message.content[0].text

@mcp.tool()
async def claude_code_review(code: str, language: str = "python") -> dict:
    """Get code review from Claude."""
    
    message = await claude.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=2000,
        system="You are an expert code reviewer. Provide constructive feedback on code quality, potential bugs, and improvements.",
        messages=[{
            "role": "user",
            "content": f"Review this {language} code:\n```{language}\n{code}\n```"
        }]
    )
    
    return {
        "review": message.content[0].text,
        "language": language,
        "model": "claude-3-5-sonnet"
    }
```

## OpenAPI/Swagger Integration

### Automatic Server Generation
```python
import httpx
from fastmcp import FastMCP
from fastmcp.server.openapi import RouteMap, MCPType

async def create_openapi_server():
    # Fetch OpenAPI spec
    spec_response = await httpx.get("https://api.example.com/openapi.json")
    spec = spec_response.json()
    
    # Configure authenticated client
    client = httpx.AsyncClient(
        base_url="https://api.example.com",
        headers={
            "Authorization": f"Bearer {os.getenv('API_KEY')}",
            "Content-Type": "application/json"
        },
        timeout=30.0
    )
    
    # Define custom route mappings
    route_maps = [
        # Users endpoints
        RouteMap(
            pattern=r"/users/\{user_id\}",
            methods=["GET"],
            mcp_type=MCPType.RESOURCE_TEMPLATE
        ),
        RouteMap(
            pattern=r"/users",
            methods=["GET"],
            mcp_type=MCPType.RESOURCE
        ),
        RouteMap(
            pattern=r"/users",
            methods=["POST"],
            mcp_type=MCPType.TOOL
        ),
        
        # Exclude admin endpoints
        RouteMap(
            pattern=r"/admin/.*",
            mcp_type=MCPType.EXCLUDE
        )
    ]
    
    # Create server with customization
    mcp = FastMCP.from_openapi(
        openapi_spec=spec,
        client=client,
        route_maps=route_maps,
        name="API Gateway",
        mcp_component_fn=customize_component
    )
    
    return mcp

def customize_component(component, route_info):
    """Customize generated components."""
    # Improve naming
    if component.name.startswith("get_"):
        component.name = component.name[4:]  # Remove 'get_' prefix
    
    # Add tags based on path
    if "/users/" in route_info.path:
        component.tags = ["users"]
    elif "/products/" in route_info.path:
        component.tags = ["products"]
    
    # Enhance descriptions
    if not component.description:
        method = route_info.method.upper()
        path = route_info.path
        component.description = f"{method} {path}"
    
    return component

# Run server
mcp = asyncio.run(create_openapi_server())
mcp.run()
```

### Multiple API Integration
```python
from fastmcp import FastMCP

async def create_multi_api_server():
    mcp = FastMCP("multi-api-gateway")
    
    # API 1: User Service
    user_spec = await fetch_spec("https://users.api.com/openapi.json")
    user_client = create_client("https://users.api.com", "USER_API_KEY")
    user_server = FastMCP.from_openapi(user_spec, user_client)
    
    # API 2: Product Service
    product_spec = await fetch_spec("https://products.api.com/openapi.json")
    product_client = create_client("https://products.api.com", "PRODUCT_API_KEY")
    product_server = FastMCP.from_openapi(product_spec, product_client)
    
    # Mount sub-servers with prefixes
    mcp.mount(user_server, prefix="users")
    mcp.mount(product_server, prefix="products")
    
    # Add orchestration tool
    @mcp.tool()
    async def get_user_with_products(user_id: str) -> dict:
        """Get user details with their products."""
        user = await user_server.call_tool("get_user", {"id": user_id})
        products = await product_server.call_tool("list_user_products", {"user_id": user_id})
        
        return {
            "user": user,
            "products": products
        }
    
    return mcp
```

## FastAPI Integration

### Convert FastAPI to MCP
```python
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel

# Existing FastAPI app
app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    description: str = None

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    return {"item_id": item_id, "name": "Example Item"}

@app.post("/items")
async def create_item(item: Item):
    return {"status": "created", "item": item}

# Convert to MCP
mcp = FastMCP.from_fastapi(
    app=app,
    name="FastAPI Server",
    route_maps=[
        RouteMap(
            methods=["GET"],
            pattern=r".*\{.*\}.*",
            mcp_type=MCPType.RESOURCE_TEMPLATE
        ),
        RouteMap(
            methods=["POST"],
            mcp_type=MCPType.TOOL
        )
    ]
)

# Add MCP-specific tools
@mcp.tool()
async def batch_create_items(items: list[Item]) -> dict:
    """Create multiple items at once."""
    results = []
    for item in items:
        result = await create_item(item)
        results.append(result)
    return {"created": len(results), "items": results}

if __name__ == "__main__":
    mcp.run()  # Run as MCP server
    # Or run FastAPI: uvicorn.run(app)
```

## REST API Connection

### Generic REST Client
```python
from fastmcp import FastMCP
import httpx
from typing import Any, Dict, Optional

mcp = FastMCP("rest-client")

class RESTClient:
    def __init__(self, base_url: str, headers: dict = None):
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers or {},
            timeout=30.0
        )
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        json: dict = None,
        headers: dict = None
    ) -> dict:
        """Make REST API request."""
        response = await self.client.request(
            method=method,
            url=endpoint,
            params=params,
            json=json,
            headers=headers
        )
        response.raise_for_status()
        return response.json()

# Create clients for different APIs
clients = {
    "github": RESTClient(
        "https://api.github.com",
        {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
    ),
    "stripe": RESTClient(
        "https://api.stripe.com/v1",
        {"Authorization": f"Bearer {os.getenv('STRIPE_KEY')}"}
    )
}

@mcp.tool()
async def api_request(
    service: str,
    method: str,
    endpoint: str,
    params: dict = None,
    body: dict = None
) -> dict:
    """Make request to configured REST API."""
    if service not in clients:
        return {"error": f"Unknown service: {service}"}
    
    try:
        result = await clients[service].request(
            method=method,
            endpoint=endpoint,
            params=params,
            json=body
        )
        return {"success": True, "data": result}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}", "details": str(e)}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def github_create_issue(repo: str, title: str, body: str, labels: list = None) -> dict:
    """Create GitHub issue."""
    return await api_request(
        service="github",
        method="POST",
        endpoint=f"/repos/{repo}/issues",
        body={
            "title": title,
            "body": body,
            "labels": labels or []
        }
    )
```

## Authentication Methods

### OAuth2 Flow
```python
from fastmcp import FastMCP
from fastmcp.server.auth import OAuth2Provider
import httpx

# Configure OAuth2
oauth = OAuth2Provider(
    client_id=os.getenv("OAUTH_CLIENT_ID"),
    client_secret=os.getenv("OAUTH_CLIENT_SECRET"),
    authorize_url="https://accounts.example.com/oauth/authorize",
    token_url="https://accounts.example.com/oauth/token",
    redirect_uri="http://localhost:8000/callback",
    scopes=["read", "write"]
)

mcp = FastMCP("oauth-server", auth=oauth)

# Token management
class TokenManager:
    def __init__(self):
        self.tokens = {}
    
    async def get_token(self, user_id: str) -> str:
        """Get or refresh token for user."""
        if user_id not in self.tokens:
            # Initial auth flow
            token = await oauth.get_access_token(user_id)
            self.tokens[user_id] = token
        else:
            # Check if expired and refresh
            token = self.tokens[user_id]
            if await self.is_expired(token):
                token = await oauth.refresh_token(token["refresh_token"])
                self.tokens[user_id] = token
        
        return token["access_token"]

token_manager = TokenManager()

@mcp.tool()
async def authenticated_request(user_id: str, endpoint: str) -> dict:
    """Make authenticated request for user."""
    token = await token_manager.get_token(user_id)
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.example.com/{endpoint}",
            headers={"Authorization": f"Bearer {token}"}
        )
        return response.json()
```

### API Key Authentication
```python
from fastmcp import FastMCP

mcp = FastMCP("api-key-server")

# Multiple API key management
API_KEYS = {
    "service1": os.getenv("SERVICE1_API_KEY"),
    "service2": os.getenv("SERVICE2_API_KEY"),
    "service3": os.getenv("SERVICE3_API_KEY")
}

@mcp.tool()
async def multi_service_request(
    service: str,
    endpoint: str,
    method: str = "GET",
    data: dict = None
) -> dict:
    """Request with service-specific API key."""
    
    if service not in API_KEYS:
        return {"error": f"Unknown service: {service}"}
    
    headers = {
        "X-API-Key": API_KEYS[service],
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=method,
            url=f"https://{service}.api.com/{endpoint}",
            headers=headers,
            json=data
        )
        return response.json()
```

### JWT/Bearer Token
```python
from fastmcp.server.auth import BearerAuthProvider
import jwt

# JWT validation
auth = BearerAuthProvider(
    jwks_uri="https://auth.example.com/.well-known/jwks.json",
    issuer="https://auth.example.com",
    audience="my-mcp-server",
    algorithms=["RS256"]
)

mcp = FastMCP("jwt-server", auth=auth)

@mcp.tool()
async def protected_operation(data: dict) -> dict:
    """Operation requiring valid JWT."""
    # Auth is automatically validated by BearerAuthProvider
    return {"status": "success", "data": data}

# Custom JWT generation for outgoing requests
def generate_jwt(payload: dict) -> str:
    """Generate JWT for API calls."""
    return jwt.encode(
        payload,
        os.getenv("JWT_SECRET"),
        algorithm="HS256"
    )

@mcp.tool()
async def call_with_jwt(endpoint: str, data: dict) -> dict:
    """Make API call with generated JWT."""
    token = generate_jwt({
        "sub": "mcp-server",
        "exp": time.time() + 3600,
        "data": data
    })
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.example.com/{endpoint}",
            headers={"Authorization": f"Bearer {token}"},
            json=data
        )
        return response.json()
```

## Permit.io Integration

### Authorization with Permit
```python
from fastmcp import FastMCP
from permit import Permit

# Initialize Permit
permit = Permit(
    token=os.getenv("PERMIT_API_KEY"),
    pdp="https://cloudpdp.api.permit.io"
)

mcp = FastMCP("permit-server")

async def check_permission(user: str, action: str, resource: str) -> bool:
    """Check if user has permission."""
    return await permit.check(user, action, resource)

@mcp.tool()
async def protected_read(user_id: str, resource_id: str) -> dict:
    """Read resource with permission check."""
    
    # Check permission
    if not await check_permission(user_id, "read", f"document:{resource_id}"):
        return {"error": "Permission denied"}
    
    # Fetch resource
    data = await fetch_resource(resource_id)
    return {"data": data}

@mcp.tool()
async def protected_write(user_id: str, resource_id: str, data: dict) -> dict:
    """Write resource with permission check."""
    
    # Check permission
    if not await check_permission(user_id, "write", f"document:{resource_id}"):
        return {"error": "Permission denied"}
    
    # Update resource
    result = await update_resource(resource_id, data)
    return {"status": "updated", "result": result}

@mcp.tool()
async def grant_permission(
    admin_id: str,
    user_id: str,
    permission: str,
    resource: str
) -> dict:
    """Grant permission (admin only)."""
    
    # Check admin permission
    if not await check_permission(admin_id, "admin", "permissions"):
        return {"error": "Admin access required"}
    
    # Grant permission via Permit API
    await permit.api.assign_role(user_id, permission, resource)
    
    return {
        "status": "granted",
        "user": user_id,
        "permission": permission,
        "resource": resource
    }
```

## Integration Best Practices

### 1. Error Handling
```python
from functools import wraps

def handle_integration_errors(func):
    """Decorator for consistent error handling."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            return {
                "error": "HTTP Error",
                "status": e.response.status_code,
                "details": e.response.text
            }
        except httpx.TimeoutException:
            return {"error": "Request timeout"}
        except Exception as e:
            return {"error": str(e)}
    return wrapper

@mcp.tool()
@handle_integration_errors
async def safe_api_call(endpoint: str) -> dict:
    """API call with error handling."""
    # Your API call here
    pass
```

### 2. Rate Limiting
```python
from asyncio import Semaphore

# Global rate limiter
rate_limit = Semaphore(10)  # Max 10 concurrent requests

@mcp.tool()
async def rate_limited_call(endpoint: str) -> dict:
    """API call with rate limiting."""
    async with rate_limit:
        return await make_api_call(endpoint)
```

### 3. Caching Responses
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cache_key(*args, **kwargs):
    """Generate cache key."""
    key = f"{args}:{kwargs}"
    return hashlib.md5(key.encode()).hexdigest()

cache = {}

@mcp.tool()
async def cached_api_call(endpoint: str, cache_ttl: int = 300) -> dict:
    """API call with caching."""
    key = cache_key(endpoint)
    
    # Check cache
    if key in cache:
        cached_data, timestamp = cache[key]
        if time.time() - timestamp < cache_ttl:
            return {"data": cached_data, "cached": True}
    
    # Fetch fresh
    data = await fetch_from_api(endpoint)
    cache[key] = (data, time.time())
    
    return {"data": data, "cached": False}
```

## Testing Integrations

```python
import pytest
from fastmcp.testing import create_test_client

@pytest.mark.asyncio
async def test_api_integration():
    """Test API integration."""
    async with create_test_client("server.py") as client:
        # Test tool listing
        tools = await client.list_tools()
        assert len(tools) > 0
        
        # Test API call
        result = await client.call_tool(
            "api_request",
            {
                "service": "github",
                "method": "GET",
                "endpoint": "/user"
            }
        )
        assert "data" in result or "error" in result
```

## WorkOS Authentication

WorkOS provides enterprise-grade authentication, user management, and SSO capabilities that can be integrated with FastMCP servers for secure enterprise applications.

### Server Setup with WorkOS

```python
from fastmcp import FastMCP
import workos
from fastmcp.server.auth import WorkOSAuthProvider

# Initialize WorkOS
workos.api_key = os.getenv("WORKOS_API_KEY")
workos.client_id = os.getenv("WORKOS_CLIENT_ID")

# Create WorkOS auth provider
auth_provider = WorkOSAuthProvider(
    client_id=os.getenv("WORKOS_CLIENT_ID"),
    client_secret=os.getenv("WORKOS_CLIENT_SECRET"),
    redirect_uri="http://localhost:8000/callback"
)

mcp = FastMCP("workos-server", auth=auth_provider)

@mcp.tool()
async def get_user_profile(user_id: str) -> dict:
    """Get user profile from WorkOS."""
    try:
        user = workos.user_management.get_user(user_id)
        return {
            "id": user.id,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_organization_users(organization_id: str) -> dict:
    """List users in an organization."""
    try:
        users = workos.user_management.list_users(
            organization_id=organization_id
        )
        return {
            "users": [
                {
                    "id": user.id,
                    "email": user.email,
                    "first_name": user.first_name,
                    "last_name": user.last_name
                }
                for user in users.data
            ],
            "total": len(users.data)
        }
    except Exception as e:
        return {"error": str(e)}
```

### SSO Authentication Flow

```python
from fastmcp.server.auth.workos import WorkOSSSO

# Configure SSO providers
sso = WorkOSSSO(
    client_id=os.getenv("WORKOS_CLIENT_ID"),
    client_secret=os.getenv("WORKOS_CLIENT_SECRET"),
    redirect_uri="http://localhost:8000/sso/callback"
)

@mcp.tool()
async def initiate_sso(organization_id: str, provider: str = "GoogleOAuth") -> dict:
    """Initiate SSO login for an organization."""
    try:
        authorization_url = workos.sso.get_authorization_url(
            connection_id=None,
            organization_id=organization_id,
            redirect_uri=sso.redirect_uri,
            state={"organization_id": organization_id}
        )
        
        return {
            "authorization_url": authorization_url,
            "organization_id": organization_id,
            "provider": provider
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def complete_sso_login(code: str, state: dict) -> dict:
    """Complete SSO login and get user profile."""
    try:
        # Exchange code for profile
        profile = workos.sso.get_profile_and_token(code)
        
        # Get user information
        user_info = {
            "id": profile.profile.id,
            "connection_id": profile.profile.connection_id,
            "organization_id": profile.profile.organization_id,
            "connection_type": profile.profile.connection_type,
            "email": profile.profile.email,
            "first_name": profile.profile.first_name,
            "last_name": profile.profile.last_name,
            "groups": profile.profile.groups or []
        }
        
        return {
            "user": user_info,
            "access_token": profile.access_token,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}
```

### Organization Management

```python
@mcp.tool()
async def list_organizations() -> dict:
    """List all organizations."""
    try:
        organizations = workos.organizations.list_organizations()
        return {
            "organizations": [
                {
                    "id": org.id,
                    "name": org.name,
                    "domains": org.domains,
                    "created_at": org.created_at,
                    "updated_at": org.updated_at
                }
                for org in organizations.data
            ],
            "total": len(organizations.data)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def create_organization(name: str, domains: list[str] = None) -> dict:
    """Create a new organization."""
    try:
        organization = workos.organizations.create_organization(
            name=name,
            domains=domains or []
        )
        
        return {
            "id": organization.id,
            "name": organization.name,
            "domains": organization.domains,
            "created_at": organization.created_at,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}

@mcp.tool()
async def get_organization(organization_id: str) -> dict:
    """Get organization details."""
    try:
        organization = workos.organizations.get_organization(organization_id)
        return {
            "id": organization.id,
            "name": organization.name,
            "domains": organization.domains,
            "created_at": organization.created_at,
            "updated_at": organization.updated_at
        }
    except Exception as e:
        return {"error": str(e)}
```

### Directory Sync Integration

```python
@mcp.tool()
async def list_directory_users(directory_id: str) -> dict:
    """List users from a directory sync."""
    try:
        users = workos.directory_sync.list_users(directory_id)
        return {
            "users": [
                {
                    "id": user.id,
                    "username": user.username,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "emails": [email.value for email in user.emails],
                    "state": user.state,
                    "created_at": user.created_at,
                    "updated_at": user.updated_at
                }
                for user in users.data
            ],
            "total": len(users.data)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def list_directory_groups(directory_id: str) -> dict:
    """List groups from a directory sync."""
    try:
        groups = workos.directory_sync.list_groups(directory_id)
        return {
            "groups": [
                {
                    "id": group.id,
                    "name": group.name,
                    "created_at": group.created_at,
                    "updated_at": group.updated_at
                }
                for group in groups.data
            ],
            "total": len(groups.data)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def get_directory_user(directory_id: str, user_id: str) -> dict:
    """Get specific user from directory."""
    try:
        user = workos.directory_sync.get_user(user_id)
        return {
            "id": user.id,
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "emails": [email.value for email in user.emails],
            "groups": [group.id for group in user.groups] if user.groups else [],
            "state": user.state,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
    except Exception as e:
        return {"error": str(e)}
```

### Magic Auth (Passwordless)

```python
@mcp.tool()
async def send_magic_link(email: str) -> dict:
    """Send passwordless magic link to user."""
    try:
        magic_auth = workos.user_management.send_magic_auth_code(
            email=email
        )
        
        return {
            "user_id": magic_auth.user_id,
            "email": email,
            "sent": True,
            "expires_at": magic_auth.expires_at
        }
    except Exception as e:
        return {"error": str(e), "sent": False}

@mcp.tool()
async def verify_magic_code(code: str, email: str) -> dict:
    """Verify magic auth code and authenticate user."""
    try:
        user = workos.user_management.authenticate_with_magic_auth(
            code=code,
            email=email
        )
        
        return {
            "user_id": user.user.id,
            "email": user.user.email,
            "access_token": user.access_token,
            "refresh_token": user.refresh_token,
            "authenticated": True
        }
    except Exception as e:
        return {"error": str(e), "authenticated": False}
```

### Multi-Factor Authentication (MFA)

```python
@mcp.tool()
async def enroll_mfa_factor(user_id: str, type: str = "totp") -> dict:
    """Enroll user in MFA."""
    try:
        factor = workos.mfa.enroll_factor(
            user_id=user_id,
            type=type
        )
        
        return {
            "id": factor.id,
            "type": factor.type,
            "totp": {
                "qr_code": factor.totp.qr_code,
                "secret": factor.totp.secret,
                "uri": factor.totp.uri
            } if factor.totp else None,
            "created_at": factor.created_at
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def challenge_mfa_factor(
    authentication_factor_id: str,
    authentication_challenge_id: str = None
) -> dict:
    """Create MFA challenge."""
    try:
        challenge = workos.mfa.challenge_factor(
            authentication_factor_id=authentication_factor_id
        )
        
        return {
            "id": challenge.id,
            "created_at": challenge.created_at,
            "updated_at": challenge.updated_at,
            "expires_at": challenge.expires_at
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def verify_mfa_challenge(challenge_id: str, code: str) -> dict:
    """Verify MFA challenge with code."""
    try:
        challenge = workos.mfa.verify_challenge(
            authentication_challenge_id=challenge_id,
            code=code
        )
        
        return {
            "challenge_id": challenge.id,
            "valid": challenge.valid,
            "verified": True
        }
    except Exception as e:
        return {"error": str(e), "verified": False}
```

### Audit Logs Integration

```python
@mcp.tool()
async def create_audit_log_event(
    organization_id: str,
    event_type: str,
    actor: dict,
    targets: list[dict],
    context: dict = None,
    occurred_at: str = None
) -> dict:
    """Create audit log event."""
    try:
        event = workos.audit_logs.create_event(
            organization_id=organization_id,
            event=workos.audit_logs.AuditLogEvent(
                action=event_type,
                actor_type="user",
                actor_id=actor.get("id"),
                actor_name=actor.get("name"),
                targets=targets,
                context=context or {},
                occurred_at=occurred_at
            )
        )
        
        return {
            "event_id": event.id,
            "created": True,
            "organization_id": organization_id
        }
    except Exception as e:
        return {"error": str(e), "created": False}

@mcp.tool()
async def export_audit_logs(
    organization_id: str,
    range_start: str,
    range_end: str,
    actions: list[str] = None
) -> dict:
    """Export audit logs for date range."""
    try:
        csv_url = workos.audit_logs.create_export(
            organization_id=organization_id,
            range_start=range_start,
            range_end=range_end,
            actions=actions or []
        )
        
        return {
            "export_url": csv_url,
            "organization_id": organization_id,
            "range_start": range_start,
            "range_end": range_end,
            "success": True
        }
    except Exception as e:
        return {"error": str(e), "success": False}
```

### Webhooks Integration

```python
@mcp.tool()
async def verify_webhook_signature(
    payload: str,
    signature: str,
    secret: str
) -> dict:
    """Verify WorkOS webhook signature."""
    try:
        is_valid = workos.webhooks.verify_event(
            payload=payload,
            signature=signature,
            secret=secret
        )
        
        return {
            "valid": is_valid,
            "verified": True
        }
    except Exception as e:
        return {"error": str(e), "verified": False}

@mcp.tool()
async def process_webhook_event(event_data: dict) -> dict:
    """Process incoming WorkOS webhook event."""
    try:
        event_type = event_data.get("event")
        data = event_data.get("data", {})
        
        # Handle different event types
        if event_type == "user.created":
            return await handle_user_created(data)
        elif event_type == "user.updated":
            return await handle_user_updated(data)
        elif event_type == "organization.created":
            return await handle_organization_created(data)
        elif event_type == "directory_sync.user.created":
            return await handle_directory_user_created(data)
        else:
            return {"event": event_type, "processed": False, "reason": "Unhandled event type"}
            
    except Exception as e:
        return {"error": str(e), "processed": False}

async def handle_user_created(user_data: dict) -> dict:
    """Handle user created event."""
    # Your user creation logic here
    return {
        "event": "user.created",
        "user_id": user_data.get("id"),
        "processed": True
    }

async def handle_user_updated(user_data: dict) -> dict:
    """Handle user updated event."""
    # Your user update logic here
    return {
        "event": "user.updated", 
        "user_id": user_data.get("id"),
        "processed": True
    }

async def handle_organization_created(org_data: dict) -> dict:
    """Handle organization created event."""
    # Your organization creation logic here
    return {
        "event": "organization.created",
        "organization_id": org_data.get("id"),
        "processed": True
    }

async def handle_directory_user_created(user_data: dict) -> dict:
    """Handle directory user created event."""
    # Your directory user creation logic here
    return {
        "event": "directory_sync.user.created",
        "user_id": user_data.get("id"),
        "processed": True
    }
```

### Environment Configuration

```python
# Environment variables for WorkOS integration
import os

WORKOS_CONFIG = {
    "api_key": os.getenv("WORKOS_API_KEY"),
    "client_id": os.getenv("WORKOS_CLIENT_ID"),  
    "client_secret": os.getenv("WORKOS_CLIENT_SECRET"),
    "webhook_secret": os.getenv("WORKOS_WEBHOOK_SECRET"),
    "redirect_uri": os.getenv("WORKOS_REDIRECT_URI", "http://localhost:8000/callback"),
    "environment": os.getenv("WORKOS_ENVIRONMENT", "staging")  # staging or production
}

# Validate required configuration
required_vars = ["api_key", "client_id", "client_secret"]
missing_vars = [var for var in required_vars if not WORKOS_CONFIG[var]]

if missing_vars:
    raise ValueError(f"Missing required WorkOS environment variables: {missing_vars}")
```

### Testing WorkOS Integration

```python
import pytest
from fastmcp.testing import create_test_client

@pytest.mark.asyncio
async def test_workos_user_profile():
    """Test WorkOS user profile retrieval."""
    async with create_test_client("workos_server.py") as client:
        result = await client.call_tool("get_user_profile", {"user_id": "user_123"})
        
        if "error" not in result:
            assert "id" in result
            assert "email" in result
        else:
            # Handle expected errors in test environment
            assert result["error"] is not None

@pytest.mark.asyncio
async def test_workos_sso_flow():
    """Test SSO initiation."""
    async with create_test_client("workos_server.py") as client:
        result = await client.call_tool(
            "initiate_sso", 
            {
                "organization_id": "org_123",
                "provider": "GoogleOAuth"
            }
        )
        
        if "error" not in result:
            assert "authorization_url" in result
            assert "organization_id" in result
```

## Resources

- [FastMCP OpenAPI Docs](https://gofastmcp.com/integrations/openapi)
- [Gemini SDK Guide](https://gofastmcp.com/integrations/gemini)
- [OpenAI Integration](https://gofastmcp.com/integrations/openai)
- [Anthropic Integration](https://gofastmcp.com/integrations/anthropic)
- [FastAPI Integration](https://gofastmcp.com/integrations/fastapi)
- [Permit.io Docs](https://docs.permit.io)
- [WorkOS Documentation](https://workos.com/docs)
- [WorkOS Python SDK](https://github.com/workos/workos-python)