# FastMCP v2.0.0 Migration Guide

## Overview

This guide provides step-by-step instructions for migrating MCP servers from FastMCP v1.x to v2.0.0, based on real-world experience upgrading the SimPro MCP servers.

## Breaking Changes

### 1. FastMCP Constructor
```python
# v1.0.0
mcp = FastMCP(
    name="my-server",
    version="1.0.0",
    description="Server description"  # REMOVED in v2.0.0
)

# v2.0.0
mcp = FastMCP(
    name="my-server",
    version="2.0.0"
    # description parameter no longer supported
)
```

### 2. Context API
```python
# v1.0.0
from fastmcp import Context
ctx = Context()  # Would fail in some cases

# v2.0.0
from fastmcp import Context
ctx = Context(mcp)  # Pass the FastMCP instance
# OR in tool functions, context is provided:
async def my_tool(ctx: Context, param: str):
    await ctx.report_progress(0, 100, "Starting...")
```

### 3. Resource Registration
```python
# v1.0.0 - Static resources
@mcp.resource("my://resource")
async def my_resource():
    return {"data": "value"}

# v2.0.0 - Dynamic resource templates
@mcp.resource("my://resource/{id}")
async def my_resource(id: str):
    return {
        "uri": f"my://resource/{id}",
        "name": f"Resource {id}",
        "mimeType": "application/json",
        "text": str(data)
    }
```

## New Features

### 1. Progress Reporting

Report progress for long-running operations:

```python
async def batch_process(ctx: Context, items: list):
    total = len(items)
    for i, item in enumerate(items):
        await ctx.report_progress(i, total, f"Processing {item}")
        # Process item...
```

### 2. Dynamic Resource Templates

Create parameterized resource URIs:

```python
@mcp.resource("data://items/{category}/{id}")
async def get_item(category: str, id: str):
    data = await fetch_item(category, id)
    return {
        "uri": f"data://items/{category}/{id}",
        "name": f"{category.title()} Item {id}",
        "mimeType": "application/json",
        "text": json.dumps(data)
    }
```

### 3. Interactive Workflows

Use `elicit()` for interactive user input:

```python
from fastmcp import elicit

@mcp.tool()
async def interactive_setup():
    name = await elicit("What's your name?", str)
    age = await elicit("What's your age?", int)
    confirmed = await elicit("Confirm setup?", bool)
    
    if confirmed:
        return {"name": name, "age": age}
```

## Migration Steps

### Step 1: Update Dependencies

```bash
# Update requirements.txt
fastmcp>=2.0.0
pydantic>=2.0.0

# Install updates
pip install -r requirements.txt --upgrade
```

### Step 2: Update Server Initialization

```python
# server.py
from fastmcp import FastMCP

# Remove description parameter
mcp = FastMCP(
    name="my-server",
    version="2.0.0"  # Update version
)
```

### Step 3: Fix Import Paths

```python
# Change relative imports to absolute
# Before:
from ..shared import Config

# After:
from shared import Config
```

### Step 4: Update Tool Signatures

```python
# Add Context parameter if using progress reporting
@mcp.tool()
async def my_tool(ctx: Context, param: str):
    # Can now use ctx.report_progress()
    await ctx.report_progress(0, 1, "Starting...")
    result = await process(param)
    await ctx.report_progress(1, 1, "Complete")
    return result
```

### Step 5: Convert Resources to Templates

```python
# Before: Static resources
@mcp.resource("customer://list")
async def customer_list():
    return {"customers": [...]}

# After: Dynamic templates
@mcp.resource("customer://list/{status}")
async def customer_list(status: str = "active"):
    customers = await fetch_customers(status)
    return {
        "uri": f"customer://list/{status}",
        "name": f"Customer List - {status}",
        "mimeType": "application/json",
        "text": json.dumps(customers)
    }
```

### Step 6: Reorganize Code Structure

Each server should be self-contained with its own utility modules:

```
my-server/
├── src/
│   ├── server.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── domain1.py
│   │   ├── domain2.py
│   │   └── domain3.py
│   ├── resources/
│   │   ├── __init__.py
│   │   └── templates.py
│   └── shared/          # Self-contained utilities
│       ├── __init__.py
│       ├── config.py
│       ├── api_client.py
│       ├── resilience.py
│       └── monitoring.py
└── requirements.txt
```

### Step 7: Add Type Annotations

```python
# Ensure all Pydantic models have annotations
from pydantic import BaseModel
from typing import Optional

class MyModel(BaseModel):
    name: str  # Required annotation
    value: Optional[int] = None  # Optional with default
```

### Step 8: Implement Error Handling

```python
# Add retry logic for resilience
async def resilient_api_call(func, *args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await func(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
```

### Step 9: Create Validation Script

```python
# validate.py
def validate_server():
    try:
        # Test imports
        from server import mcp
        from tools import TOOL_METADATA
        
        # Check registration
        assert len(TOOL_METADATA) > 0, "No tools registered"
        
        # Test resource templates
        # ...
        
        print("✓ Validation passed")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

if __name__ == "__main__":
    validate_server()
```

### Step 10: Update Documentation

Update your README and API documentation to reflect:
- New v2.0.0 features
- Changed resource URIs
- Progress reporting capability
- Interactive workflow support

## Common Issues and Solutions

### Issue 1: ImportError for relative imports
**Solution:** Change to absolute imports without `..`

### Issue 2: PydanticUserError for non-annotated attributes
**Solution:** Add type annotations to all model attributes

### Issue 3: Context initialization error
**Solution:** Pass mcp instance or use provided context in tools

### Issue 4: Resource templates not working
**Solution:** Ensure return dict includes all required fields (uri, name, mimeType, text)

### Issue 5: Progress reporting not visible
**Solution:** Ensure Context is properly passed and client supports progress

## Testing Checklist

- [ ] All imports resolve correctly
- [ ] Server starts without errors
- [ ] Tools are registered and callable
- [ ] Resource templates return correct URIs
- [ ] Progress reporting works for batch operations
- [ ] Error handling and retry logic function
- [ ] Interactive workflows prompt correctly
- [ ] All tests pass

## Performance Considerations

### Before Migration
- Consider current bottlenecks
- Identify batch operations needing progress
- List resources that could be dynamic

### After Migration
- Monitor progress reporting overhead
- Test resource template caching
- Validate retry logic timing
- Check memory usage with new features

## Rollback Plan

If issues arise:

1. **Keep v1.0.0 branch available**
```bash
git branch backup-v1.0.0
git checkout -b feature/v2-migration
```

2. **Test in isolation**
- Run v2.0.0 server on different port
- Validate with subset of operations
- Compare outputs with v1.0.0

3. **Gradual rollout**
- Migrate one tool category at a time
- Test thoroughly before proceeding
- Keep v1.0.0 server running in parallel

## Best Practices

1. **Use Type Hints**
```python
async def process_items(
    ctx: Context,
    items: List[Dict[str, Any]],
    options: Optional[ProcessOptions] = None
) -> ProcessResult:
```

2. **Implement Progress Reporting**
```python
async def long_operation(ctx: Context, data: list):
    steps = ["Validating", "Processing", "Finalizing"]
    for i, step in enumerate(steps):
        await ctx.report_progress(i, len(steps), step)
        await perform_step(step, data)
```

3. **Self-Contained Architecture**
```python
# Each server contains its own copy of utility modules
# shared/patterns.py (duplicated across servers)
async def batch_with_progress(ctx, items, processor):
    results = []
    for i, item in enumerate(items):
        await ctx.report_progress(i, len(items))
        result = await processor(item)
        results.append(result)
    return results
```

4. **Document Resource Templates**
```python
@mcp.resource("data://report/{type}/{period}")
async def report_resource(type: str, period: str):
    """
    Generate report resource.
    
    Args:
        type: Report type (sales, inventory, performance)
        period: Time period (daily, weekly, monthly)
    
    Returns:
        Resource with report data
    """
```

## Conclusion

Migrating to FastMCP v2.0.0 provides significant benefits:
- Better user feedback through progress reporting
- More flexible resource access patterns
- Interactive workflow capabilities
- Enhanced error resilience

Follow this guide systematically, test thoroughly, and maintain a rollback plan for a smooth migration.

---

*Based on SimPro MCP Servers migration experience*  
*Last updated: 2025-08-22*