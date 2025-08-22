# FastMCP Self-Hosting Guide

> **Note**: This guide covers self-hosting FastMCP servers. For most users, we recommend using [FastMCP Cloud](https://fastmcp.cloud) for simpler deployment and management.

## Docker Deployment

### Basic Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "fastmcp", "run", "server.py"]
```

### Multi-Stage Docker Build
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "fastmcp", "run", "server.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  mcp-server:
    build: .
    ports:
      - "5000:5000"
    environment:
      - API_KEY=${API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

### Docker Commands
```bash
# Generate Dockerfile
fastmcp docker generate

# Build Docker image
fastmcp docker build --tag my-server

# Run in Docker
fastmcp docker run --port 5000
```

## Systemd Service

### Service Configuration
```ini
[Unit]
Description=FastMCP Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/mcp-server
Environment="PATH=/usr/local/bin:/usr/bin"
ExecStart=/usr/local/bin/python -m fastmcp run server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Installation
```bash
# Copy service file
sudo cp mcp-server.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start service
sudo systemctl enable mcp-server
sudo systemctl start mcp-server

# Check status
sudo systemctl status mcp-server
```

## Process Managers

### PM2 (Node.js)
```json
{
  "apps": [{
    "name": "mcp-server",
    "script": "python",
    "args": ["-m", "fastmcp", "run", "server.py"],
    "cwd": "/opt/mcp-server",
    "env": {
      "API_KEY": "your-key"
    },
    "instances": 1,
    "autorestart": true,
    "watch": false,
    "max_memory_restart": "1G"
  }]
}
```

### Supervisor
```ini
[program:mcp-server]
command=/usr/local/bin/python -m fastmcp run server.py
directory=/opt/mcp-server
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/mcp-server.err.log
stdout_logfile=/var/log/mcp-server.out.log
environment=PATH="/usr/local/bin",API_KEY="your-key"
```

## Reverse Proxy Configuration

### Nginx
```nginx
server {
    listen 80;
    server_name mcp.example.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Apache
```apache
<VirtualHost *:80>
    ServerName mcp.example.com
    
    ProxyRequests Off
    ProxyPreserveHost On
    
    <Proxy *>
        Order deny,allow
        Allow from all
    </Proxy>
    
    ProxyPass / http://localhost:5000/
    ProxyPassReverse / http://localhost:5000/
    
    <Location />
        Order allow,deny
        Allow from all
    </Location>
</VirtualHost>
```

## Environment Management

### Production Environment
```bash
# /etc/environment or .env.production
NODE_ENV=production
LOG_LEVEL=info
API_KEY=prod-key-here
DATABASE_URL=postgresql://user:pass@db:5432/prod
REDIS_URL=redis://redis:6379/0
SENTRY_DSN=https://key@sentry.io/project
```

### Health Checks
```python
# health.py
from fastmcp import FastMCP

mcp = FastMCP("health-check")

@mcp.tool()
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@mcp.tool()
async def readiness():
    """Readiness check."""
    # Check database, cache, etc.
    checks = {
        "database": await check_database(),
        "cache": await check_cache(),
        "api": await check_external_api()
    }
    
    all_ready = all(checks.values())
    return {
        "ready": all_ready,
        "checks": checks
    }
```

## Monitoring

### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest

request_count = Counter('mcp_requests_total', 'Total requests')
request_duration = Histogram('mcp_request_duration_seconds', 'Request duration')

@mcp.tool()
async def metrics():
    """Expose Prometheus metrics."""
    return generate_latest()
```

### Logging Configuration
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            'logs/mcp-server.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)
```

## Security Considerations

### SSL/TLS
- Always use HTTPS in production
- Configure SSL certificates (Let's Encrypt recommended)
- Enable HTTP/2 for better performance

### Firewall Rules
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable
```

### Rate Limiting
```python
from fastmcp import FastMCP
from functools import wraps
import time

rate_limits = {}

def rate_limit(max_calls=100, period=60):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{request.remote_addr}"
            now = time.time()
            
            if key not in rate_limits:
                rate_limits[key] = []
            
            # Clean old entries
            rate_limits[key] = [t for t in rate_limits[key] if now - t < period]
            
            if len(rate_limits[key]) >= max_calls:
                raise Exception("Rate limit exceeded")
            
            rate_limits[key].append(now)
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

## Backup Strategy

### Database Backup
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
pg_dump $DATABASE_URL > $BACKUP_DIR/db_$DATE.sql

# Compress
gzip $BACKUP_DIR/db_$DATE.sql

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
```

### Cron Configuration
```cron
# Daily backup at 2 AM
0 2 * * * /opt/mcp-server/backup.sh

# Weekly restart Sunday at 3 AM
0 3 * * 0 systemctl restart mcp-server
```

## Troubleshooting Self-Hosted Deployments

### Common Issues

1. **Port Already in Use**
   ```bash
   lsof -i :5000
   kill -9 <PID>
   ```

2. **Permission Denied**
   ```bash
   chown -R www-data:www-data /opt/mcp-server
   chmod +x server.py
   ```

3. **Module Not Found**
   ```bash
   pip install -r requirements.txt
   python -m pip list
   ```

4. **Service Won't Start**
   ```bash
   journalctl -u mcp-server -f
   systemctl status mcp-server
   ```

## Migration from Self-Hosted to FastMCP Cloud

When ready to migrate to FastMCP Cloud:

1. **Prepare your repository**
   - Ensure module-level server object
   - Use only PyPI dependencies
   - Add environment variables to `.env`

2. **Test locally**
   ```bash
   fastmcp dev server.py
   ```

3. **Deploy to cloud**
   - Push to GitHub
   - Connect repository to FastMCP Cloud
   - Configure environment variables
   - Deploy with one click

## Additional Resources

- [Docker Documentation](https://docs.docker.com)
- [Systemd Documentation](https://systemd.io)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [PM2 Documentation](https://pm2.keymetrics.io)
- [Supervisor Documentation](http://supervisord.org)