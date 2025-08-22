# FastMCP Testing Documentation

## Table of Contents
- [Overview](#overview)
- [Test Client Setup](#test-client-setup)
- [Unit Testing Tools](#unit-testing-tools)
- [Testing Resources](#testing-resources)
- [Testing Prompts](#testing-prompts)
- [Mock Servers](#mock-servers)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Testing](#performance-testing)
- [Test Fixtures and Helpers](#test-fixtures-and-helpers)
- [CI/CD Integration](#cicd-integration)
- [Best Practices](#best-practices)

## Overview

FastMCP provides comprehensive testing utilities to ensure your MCP servers work correctly. This guide covers unit testing, integration testing, mocking, and best practices for testing MCP servers and clients.

## Test Client Setup

### Basic Test Client

```python
import pytest
from fastmcp.testing import TestClient
from fastmcp import FastMCP

# Your server
mcp = FastMCP("test-server")

@mcp.tool()
async def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Test using TestClient
@pytest.mark.asyncio
async def test_add_numbers():
    """Test the add_numbers tool."""
    async with TestClient(mcp) as client:
        result = await client.call_tool("add_numbers", a=5, b=3)
        assert result == 8
```

### Advanced Test Client Configuration

```python
from fastmcp.testing import TestClient, TestConfig

@pytest.mark.asyncio
async def test_with_config():
    """Test with custom configuration."""
    config = TestConfig(
        timeout=30,
        max_retries=3,
        mock_network=True,
        capture_logs=True
    )
    
    async with TestClient(mcp, config=config) as client:
        # Set up test context
        client.set_user({"id": "test-user", "email": "test@example.com"})
        client.set_headers({"X-Test": "true"})
        
        # Test tool call
        result = await client.call_tool("protected_tool")
        assert result["user"] == "test@example.com"
        
        # Check captured logs
        logs = client.get_logs()
        assert any("protected_tool" in log for log in logs)
```

## Unit Testing Tools

### Testing Individual Tools

```python
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastmcp import FastMCP

mcp = FastMCP("test-server")

@mcp.tool()
async def fetch_data(source: str) -> dict:
    """Fetch data from external source."""
    # In production, this would make an API call
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/{source}") as response:
            return await response.json()

@pytest.mark.asyncio
async def test_fetch_data():
    """Test fetch_data with mocked API."""
    with patch("aiohttp.ClientSession") as mock_session:
        # Set up mock response
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"data": "test"})
        
        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
        
        # Test the tool
        result = await fetch_data(source="test")
        assert result == {"data": "test"}
```

### Testing Tool Validation

```python
@mcp.tool()
async def validated_tool(
    email: str,
    age: int = None,
    tags: list[str] = None
) -> dict:
    """Tool with validation."""
    import re
    
    # Validate email
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        raise ValueError("Invalid email format")
    
    # Validate age if provided
    if age is not None and (age < 0 or age > 150):
        raise ValueError("Invalid age")
    
    return {
        "email": email,
        "age": age,
        "tags": tags or []
    }

@pytest.mark.asyncio
async def test_tool_validation():
    """Test tool input validation."""
    async with TestClient(mcp) as client:
        # Valid input
        result = await client.call_tool(
            "validated_tool",
            email="user@example.com",
            age=25,
            tags=["tag1", "tag2"]
        )
        assert result["email"] == "user@example.com"
        
        # Invalid email
        with pytest.raises(ValueError, match="Invalid email"):
            await client.call_tool(
                "validated_tool",
                email="invalid-email"
            )
        
        # Invalid age
        with pytest.raises(ValueError, match="Invalid age"):
            await client.call_tool(
                "validated_tool",
                email="user@example.com",
                age=200
            )
```

### Testing Async Tools

```python
import asyncio

@mcp.tool()
async def long_running_tool(duration: int) -> str:
    """Simulate long-running operation."""
    await asyncio.sleep(duration)
    return f"Completed after {duration} seconds"

@pytest.mark.asyncio
async def test_long_running_tool():
    """Test async tool with timeout."""
    async with TestClient(mcp) as client:
        # Test normal completion
        result = await client.call_tool("long_running_tool", duration=1)
        assert "Completed after 1" in result
        
        # Test timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                client.call_tool("long_running_tool", duration=10),
                timeout=2
            )
```

## Testing Resources

### Static Resource Testing

```python
@mcp.resource("config://app")
async def app_config() -> dict:
    """Get application configuration."""
    return {
        "version": "1.0.0",
        "features": ["auth", "api", "cache"],
        "environment": "test"
    }

@pytest.mark.asyncio
async def test_static_resource():
    """Test static resource retrieval."""
    async with TestClient(mcp) as client:
        result = await client.get_resource("config://app")
        assert result["version"] == "1.0.0"
        assert "auth" in result["features"]
```

### Dynamic Resource Testing

```python
@mcp.resource("user://{user_id}/profile")
async def user_profile(user_id: str) -> dict:
    """Get user profile."""
    # Simulate database lookup
    if user_id == "123":
        return {
            "id": "123",
            "name": "Test User",
            "email": "test@example.com"
        }
    raise ValueError(f"User {user_id} not found")

@pytest.mark.asyncio
async def test_dynamic_resource():
    """Test dynamic resource with parameters."""
    async with TestClient(mcp) as client:
        # Valid user
        result = await client.get_resource("user://123/profile")
        assert result["name"] == "Test User"
        
        # Invalid user
        with pytest.raises(ValueError, match="User 999 not found"):
            await client.get_resource("user://999/profile")
```

### Testing Resource Templates

```python
@mcp.resource("data://{dataset}/item/{item_id}")
async def data_item(dataset: str, item_id: str) -> dict:
    """Get item from dataset."""
    return {
        "dataset": dataset,
        "item_id": item_id,
        "data": f"Data for {item_id} in {dataset}"
    }

@pytest.mark.asyncio
async def test_resource_template():
    """Test resource template parsing."""
    async with TestClient(mcp) as client:
        result = await client.get_resource("data://products/item/abc123")
        assert result["dataset"] == "products"
        assert result["item_id"] == "abc123"
```

## Testing Prompts

### Basic Prompt Testing

```python
@mcp.prompt("greeting")
async def greeting_prompt(name: str = "World") -> str:
    """Generate greeting prompt."""
    return f"Hello, {name}! How can I help you today?"

@pytest.mark.asyncio
async def test_prompt():
    """Test prompt generation."""
    async with TestClient(mcp) as client:
        # Default parameter
        result = await client.get_prompt("greeting")
        assert result == "Hello, World! How can I help you today?"
        
        # Custom parameter
        result = await client.get_prompt("greeting", name="Alice")
        assert result == "Hello, Alice! How can I help you today?"
```

### Testing Complex Prompts

```python
@mcp.prompt("analysis")
async def analysis_prompt(
    data: list[dict],
    format: str = "summary",
    include_stats: bool = True
) -> str:
    """Generate analysis prompt."""
    prompt_parts = [f"Analyze the following {len(data)} items:"]
    
    if format == "summary":
        prompt_parts.append("Provide a brief summary.")
    elif format == "detailed":
        prompt_parts.append("Provide detailed analysis.")
    
    if include_stats:
        prompt_parts.append("Include statistical metrics.")
    
    return "\n".join(prompt_parts)

@pytest.mark.asyncio
async def test_complex_prompt():
    """Test prompt with multiple parameters."""
    async with TestClient(mcp) as client:
        data = [{"id": 1}, {"id": 2}]
        
        result = await client.get_prompt(
            "analysis",
            data=data,
            format="detailed",
            include_stats=False
        )
        
        assert "2 items" in result
        assert "detailed analysis" in result
        assert "statistical metrics" not in result
```

## Mock Servers

### Creating a Mock Server

```python
from fastmcp.testing import MockServer, MockResponse

@pytest.mark.asyncio
async def test_with_mock_server():
    """Test with mock MCP server."""
    mock_server = MockServer()
    
    # Define mock responses
    mock_server.add_tool_response(
        "calculate",
        MockResponse(result=42)
    )
    
    mock_server.add_resource_response(
        "config://db",
        MockResponse(data={"host": "localhost", "port": 5432})
    )
    
    async with mock_server:
        # Test tool call
        result = await mock_server.call_tool("calculate", x=10, y=32)
        assert result["result"] == 42
        
        # Test resource
        config = await mock_server.get_resource("config://db")
        assert config["data"]["host"] == "localhost"
        
        # Verify calls were made
        assert mock_server.tool_called("calculate")
        assert mock_server.resource_accessed("config://db")
```

### Mock Server with Behavior

```python
class BehavioralMockServer(MockServer):
    """Mock server with custom behavior."""
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
    
    async def handle_tool_call(self, name: str, **kwargs):
        """Custom tool handling."""
        self.call_count += 1
        
        if name == "rate_limited_tool":
            if self.call_count > 3:
                raise Exception("Rate limit exceeded")
            return {"count": self.call_count}
        
        return await super().handle_tool_call(name, **kwargs)

@pytest.mark.asyncio
async def test_behavioral_mock():
    """Test mock server with behavior."""
    mock = BehavioralMockServer()
    
    async with mock:
        # First three calls succeed
        for i in range(3):
            result = await mock.call_tool("rate_limited_tool")
            assert result["count"] == i + 1
        
        # Fourth call fails
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await mock.call_tool("rate_limited_tool")
```

## Integration Testing

### Testing Server with Dependencies

```python
import aioredis
from fastmcp.testing import IntegrationTestCase

class ServerIntegrationTest(IntegrationTestCase):
    """Integration tests for server with Redis."""
    
    @classmethod
    async def setup_class(cls):
        """Set up test dependencies."""
        # Start Redis container or use test instance
        cls.redis = await aioredis.create_redis_pool(
            "redis://localhost:6379/1"
        )
        
        # Create server with Redis
        cls.mcp = FastMCP("integration-server")
        cls.mcp.redis = cls.redis
        
        @cls.mcp.tool()
        async def cache_set(key: str, value: str) -> bool:
            """Set cache value."""
            await cls.mcp.redis.set(key, value)
            return True
        
        @cls.mcp.tool()
        async def cache_get(key: str) -> str:
            """Get cache value."""
            value = await cls.mcp.redis.get(key)
            return value.decode() if value else None
    
    @classmethod
    async def teardown_class(cls):
        """Clean up test dependencies."""
        await cls.redis.flushdb()
        cls.redis.close()
        await cls.redis.wait_closed()
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache operations with real Redis."""
        async with TestClient(self.mcp) as client:
            # Set value
            result = await client.call_tool("cache_set", key="test", value="data")
            assert result is True
            
            # Get value
            result = await client.call_tool("cache_get", key="test")
            assert result == "data"
            
            # Get non-existent value
            result = await client.call_tool("cache_get", key="missing")
            assert result is None
```

### Testing Multiple Servers

```python
@pytest.mark.asyncio
async def test_server_communication():
    """Test communication between multiple servers."""
    
    # Create producer server
    producer = FastMCP("producer")
    
    @producer.tool()
    async def produce_data() -> dict:
        return {"data": "test", "timestamp": time.time()}
    
    # Create consumer server
    consumer = FastMCP("consumer")
    
    @consumer.tool()
    async def consume_data(data: dict) -> str:
        return f"Processed: {data['data']}"
    
    # Test interaction
    async with TestClient(producer) as prod_client, \
               TestClient(consumer) as cons_client:
        
        # Produce data
        data = await prod_client.call_tool("produce_data")
        
        # Consume data
        result = await cons_client.call_tool("consume_data", data=data)
        assert "Processed: test" in result
```

## End-to-End Testing

### Full Workflow Testing

```python
@pytest.mark.asyncio
async def test_complete_workflow():
    """Test complete user workflow."""
    
    mcp = FastMCP("workflow-server")
    
    # Define workflow tools
    @mcp.tool()
    async def create_user(name: str, email: str) -> dict:
        return {"id": "123", "name": name, "email": email}
    
    @mcp.tool()
    async def create_project(user_id: str, title: str) -> dict:
        return {"id": "456", "user_id": user_id, "title": title}
    
    @mcp.tool()
    async def add_task(project_id: str, description: str) -> dict:
        return {"id": "789", "project_id": project_id, "description": description}
    
    async with TestClient(mcp) as client:
        # Complete workflow
        user = await client.call_tool(
            "create_user",
            name="Test User",
            email="test@example.com"
        )
        
        project = await client.call_tool(
            "create_project",
            user_id=user["id"],
            title="Test Project"
        )
        
        task = await client.call_tool(
            "add_task",
            project_id=project["id"],
            description="Test Task"
        )
        
        # Verify workflow
        assert task["project_id"] == project["id"]
        assert project["user_id"] == user["id"]
```

### Testing with Real Clients

```python
from fastmcp import FastMCP, MCPClient

@pytest.mark.asyncio
async def test_real_client_interaction():
    """Test with real MCP client."""
    
    # Start server in test mode
    server = FastMCP("test-server", test_mode=True)
    
    @server.tool()
    async def echo(message: str) -> str:
        return f"Echo: {message}"
    
    # Start server
    async with server.test_server() as server_url:
        # Connect real client
        async with MCPClient(server_url) as client:
            # List tools
            tools = await client.list_tools()
            assert "echo" in [t["name"] for t in tools]
            
            # Call tool
            result = await client.call_tool("echo", message="test")
            assert result == "Echo: test"
```

## Performance Testing

### Load Testing Tools

```python
import asyncio
import time
from statistics import mean, stdev

@pytest.mark.asyncio
async def test_tool_performance():
    """Test tool performance under load."""
    
    mcp = FastMCP("perf-server")
    
    @mcp.tool()
    async def compute(n: int) -> int:
        """Simulate computation."""
        await asyncio.sleep(0.01)  # Simulate work
        return n * 2
    
    async with TestClient(mcp) as client:
        # Warm up
        await client.call_tool("compute", n=1)
        
        # Measure performance
        times = []
        concurrency = 10
        iterations = 100
        
        async def measure_call():
            start = time.time()
            await client.call_tool("compute", n=42)
            return time.time() - start
        
        for _ in range(iterations // concurrency):
            tasks = [measure_call() for _ in range(concurrency)]
            batch_times = await asyncio.gather(*tasks)
            times.extend(batch_times)
        
        # Analyze results
        avg_time = mean(times)
        std_dev = stdev(times)
        
        print(f"Average response time: {avg_time:.3f}s")
        print(f"Standard deviation: {std_dev:.3f}s")
        print(f"Throughput: {len(times)/sum(times):.1f} req/s")
        
        # Assert performance requirements
        assert avg_time < 0.1  # Less than 100ms average
        assert std_dev < 0.05  # Consistent performance
```

### Memory Testing

```python
import tracemalloc
import gc

@pytest.mark.asyncio
async def test_memory_usage():
    """Test server memory usage."""
    
    mcp = FastMCP("memory-server")
    
    @mcp.tool()
    async def process_data(size: int) -> int:
        """Process data of given size."""
        data = [i for i in range(size)]
        return sum(data)
    
    async with TestClient(mcp) as client:
        # Start memory tracking
        tracemalloc.start()
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()
        
        # Perform operations
        for _ in range(100):
            await client.call_tool("process_data", size=1000)
        
        # Take second snapshot
        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()
        
        # Analyze memory usage
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        total_diff = sum(stat.size_diff for stat in top_stats)
        
        print(f"Memory increase: {total_diff / 1024 / 1024:.2f} MB")
        
        # Assert no memory leaks
        assert total_diff < 10 * 1024 * 1024  # Less than 10MB increase
```

## Test Fixtures and Helpers

### Pytest Fixtures

```python
import pytest
from pathlib import Path
import tempfile

@pytest.fixture
async def test_client(mcp_server):
    """Provide test client for server."""
    async with TestClient(mcp_server) as client:
        yield client

@pytest.fixture
def temp_dir():
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
async def mock_api():
    """Provide mock API server."""
    from aiohttp import web
    
    app = web.Application()
    
    async def handle_request(request):
        return web.json_response({"status": "ok"})
    
    app.router.add_get("/api/{path:.*}", handle_request)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()
    
    yield "http://localhost:8080"
    
    await runner.cleanup()

# Use fixtures in tests
@pytest.mark.asyncio
async def test_with_fixtures(test_client, temp_dir, mock_api):
    """Test using fixtures."""
    # Use test client
    result = await test_client.call_tool("some_tool")
    
    # Use temp directory
    test_file = temp_dir / "test.txt"
    test_file.write_text("test data")
    
    # Use mock API
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{mock_api}/api/test") as response:
            data = await response.json()
            assert data["status"] == "ok"
```

### Test Helpers

```python
from contextlib import asynccontextmanager

class TestHelpers:
    """Utility functions for testing."""
    
    @staticmethod
    async def create_test_data(count: int) -> list:
        """Create test data."""
        return [
            {"id": i, "name": f"Item {i}", "value": i * 10}
            for i in range(count)
        ]
    
    @staticmethod
    @asynccontextmanager
    async def timed_operation(name: str, max_seconds: float):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            print(f"{name} took {duration:.3f}s")
            if duration > max_seconds:
                pytest.fail(f"{name} exceeded {max_seconds}s limit")
    
    @staticmethod
    def assert_response_format(response: dict, required_fields: list):
        """Assert response has required fields."""
        for field in required_fields:
            assert field in response, f"Missing required field: {field}"
    
    @staticmethod
    async def wait_for_condition(
        condition: callable,
        timeout: float = 5.0,
        interval: float = 0.1
    ):
        """Wait for condition to become true."""
        start = time.time()
        while time.time() - start < timeout:
            if await condition():
                return True
            await asyncio.sleep(interval)
        raise TimeoutError(f"Condition not met within {timeout}s")

# Use helpers in tests
@pytest.mark.asyncio
async def test_with_helpers():
    """Test using helper functions."""
    
    # Create test data
    data = await TestHelpers.create_test_data(10)
    assert len(data) == 10
    
    # Time operation
    async with TestHelpers.timed_operation("processing", max_seconds=1.0):
        await asyncio.sleep(0.5)  # Simulate work
    
    # Check response format
    response = {"id": "123", "status": "ok", "data": {}}
    TestHelpers.assert_response_format(response, ["id", "status", "data"])
    
    # Wait for condition
    counter = 0
    async def increment():
        nonlocal counter
        counter += 1
        return counter >= 5
    
    await TestHelpers.wait_for_condition(increment)
    assert counter >= 5
```

## CI/CD Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Test MCP Server

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        env:
          REDIS_URL: redis://localhost:6379
        run: |
          pytest tests/ \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --junit-xml=test-results.xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            test-results.xml
            htmlcov/
```

### pytest.ini Configuration

```ini
# pytest.ini
[tool:pytest]
minversion = 7.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto

addopts = 
    -v
    --strict-markers
    --tb=short
    --disable-warnings
    --color=yes
    --cov=src
    --cov-branch
    --cov-report=term-missing:skip-covered

markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    e2e: marks tests as end-to-end tests

env_files =
    .env.test

log_cli = true
log_cli_level = INFO
log_cli_format = %(asctime)s [%(levelname)8s] %(message)s
log_cli_date_format = %Y-%m-%d %H:%M:%S
```

## Best Practices

### 1. Test Organization

```python
# tests/
# ├── unit/
# │   ├── test_tools.py
# │   ├── test_resources.py
# │   └── test_prompts.py
# ├── integration/
# │   ├── test_database.py
# │   ├── test_api.py
# │   └── test_cache.py
# ├── e2e/
# │   ├── test_workflows.py
# │   └── test_scenarios.py
# ├── fixtures/
# │   ├── __init__.py
# │   └── common.py
# └── conftest.py
```

### 2. Test Naming

```python
class TestUserTools:
    """Group related tests."""
    
    async def test_create_user_with_valid_data(self):
        """Descriptive test names."""
        pass
    
    async def test_create_user_with_invalid_email_raises_error(self):
        """Clear expectation in name."""
        pass
    
    async def test_create_user_without_required_fields_fails(self):
        """Describe the scenario and outcome."""
        pass
```

### 3. Test Isolation

```python
@pytest.mark.asyncio
async def test_isolated_operation():
    """Each test should be independent."""
    
    # Setup - create fresh state
    mcp = FastMCP("isolated-server")
    test_data = create_test_data()
    
    # Execute - perform operation
    result = await perform_operation(test_data)
    
    # Verify - check results
    assert result.status == "success"
    
    # Cleanup - handled by fixtures/context managers
    # No manual cleanup needed
```

### 4. Assertion Best Practices

```python
# Good: Specific assertions with clear messages
assert user["email"] == "test@example.com", "Email should match input"
assert len(results) == 5, f"Expected 5 results, got {len(results)}"

# Bad: Generic assertions
assert result  # What are we checking?
assert data == expected  # Hard to debug when it fails

# Good: Multiple specific assertions
assert response["status"] == 200
assert "data" in response
assert isinstance(response["data"], list)
assert len(response["data"]) > 0

# Bad: Complex assertion
assert response["status"] == 200 and "data" in response and len(response["data"]) > 0
```

### 5. Mocking Best Practices

```python
# Good: Mock at the boundary
@patch("external_library.api_call")
async def test_with_mock(mock_api):
    mock_api.return_value = {"status": "ok"}
    result = await our_function()
    assert result == "processed"

# Bad: Mock internal implementation
@patch("our_module.internal_function")
async def test_bad_mock(mock_internal):
    # Don't mock your own code
    pass

# Good: Use dependency injection
class Service:
    def __init__(self, api_client):
        self.api_client = api_client
    
    async def process(self):
        return await self.api_client.fetch()

# In tests
mock_client = Mock()
service = Service(mock_client)
```

### 6. Performance Test Guidelines

```python
@pytest.mark.slow
@pytest.mark.asyncio
async def test_performance():
    """Mark slow tests for optional execution."""
    
    # Run performance tests separately
    # pytest -m slow
    
    # Set reasonable timeouts
    with pytest.timeout(30):
        await long_running_test()
    
    # Use benchmarks
    # pytest-benchmark for detailed performance analysis
```

## Summary

Testing FastMCP servers requires a comprehensive approach:

1. **Unit Tests**: Test individual tools, resources, and prompts in isolation
2. **Integration Tests**: Test server with real dependencies
3. **E2E Tests**: Test complete workflows and user scenarios
4. **Performance Tests**: Ensure servers meet performance requirements
5. **Mock Servers**: Test client code without real servers
6. **Test Fixtures**: Reusable test components and helpers
7. **CI/CD**: Automated testing in continuous integration

Key principles:
- Test in isolation
- Use descriptive test names
- Mock external dependencies
- Assert specific conditions
- Organize tests logically
- Automate test execution
- Monitor test coverage

Comprehensive testing ensures your MCP servers are reliable, performant, and maintainable.