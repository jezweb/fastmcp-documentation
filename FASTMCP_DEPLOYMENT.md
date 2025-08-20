# FastMCP Deployment Documentation

## Table of Contents
- [Overview](#overview)
- [Deployment Strategies](#deployment-strategies)
- [Cloud Providers](#cloud-providers)
- [Container Deployment](#container-deployment)
- [Serverless Deployment](#serverless-deployment)
- [Traditional Server Deployment](#traditional-server-deployment)
- [CI/CD Integration](#cicd-integration)
- [Environment Configuration](#environment-configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Security Considerations](#security-considerations)
- [Scaling and Load Balancing](#scaling-and-load-balancing)
- [Troubleshooting](#troubleshooting)

## Overview

FastMCP servers can be deployed in various environments ranging from local development to large-scale production systems. This guide covers different deployment strategies, cloud provider integrations, and best practices for production deployments.

## Deployment Strategies

### 1. Local Development
```bash
# Development server with auto-reload
fastmcp dev src/server.py --debug

# Production-like local testing
fastmcp run src/server.py --host 0.0.0.0 --port 8000
```

### 2. Single Server Deployment
```bash
# Simple production deployment
fastmcp run src/server.py --transport websocket --workers 4

# With environment configuration
fastmcp run src/server.py --env-file .env.production
```

### 3. Container-Based Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["fastmcp", "run", "src/server.py", "--host", "0.0.0.0"]
```

### 4. Serverless Deployment
```python
# serverless_handler.py
import asyncio
from fastmcp import FastMCP
from fastmcp.adapters.serverless import ServerlessAdapter

mcp = FastMCP()

# Define your tools, resources, and prompts
@mcp.tool()
async def hello(name: str) -> str:
    return f"Hello, {name}!"

# Serverless handler
adapter = ServerlessAdapter(mcp)

def lambda_handler(event, context):
    return asyncio.run(adapter.handle(event, context))

# For AWS Lambda
def handler(event, context):
    return lambda_handler(event, context)

# For Google Cloud Functions
def main(request):
    return adapter.handle_http(request)

# For Azure Functions
def azure_handler(req):
    return adapter.handle_azure(req)
```

## Cloud Providers

### AWS Deployment

#### AWS Lambda
```yaml
# serverless.yml
service: mcp-server

provider:
  name: aws
  runtime: python3.11
  region: us-west-2
  environment:
    MCP_ENVIRONMENT: production
    MCP_LOG_LEVEL: INFO

functions:
  mcp:
    handler: serverless_handler.handler
    timeout: 30
    memorySize: 512
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements
```

#### AWS ECS
```json
{
  "family": "mcp-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "mcp-server",
      "image": "your-repo/mcp-server:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MCP_ENVIRONMENT", "value": "production"},
        {"name": "MCP_LOG_LEVEL", "value": "INFO"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/mcp-server",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### AWS EC2 with User Data
```bash
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git

# Clone and setup application
cd /opt
git clone https://github.com/your-org/mcp-server.git
cd mcp-server

# Install dependencies
pip3 install -r requirements.txt

# Create systemd service
cat > /etc/systemd/system/mcp-server.service << EOF
[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/mcp-server
Environment=PATH=/usr/local/bin:/usr/bin:/bin
Environment=MCP_ENVIRONMENT=production
ExecStart=/usr/local/bin/fastmcp run src/server.py --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl enable mcp-server
systemctl start mcp-server
```

### Google Cloud Platform

#### Google Cloud Functions
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - functions
      - deploy
      - mcp-server
      - --runtime=python311
      - --trigger-http
      - --entry-point=main
      - --memory=512MB
      - --timeout=60s
      - --env-vars-file=env.yaml
```

```yaml
# env.yaml
MCP_ENVIRONMENT: production
MCP_LOG_LEVEL: INFO
MCP_API_KEY: ${MCP_API_KEY}
```

#### Google Cloud Run
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: mcp-server
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/mcp-server
        ports:
        - containerPort: 8000
        env:
        - name: MCP_ENVIRONMENT
          value: production
        - name: MCP_LOG_LEVEL
          value: INFO
        resources:
          limits:
            cpu: 1000m
            memory: 512Mi
```

#### Google Kubernetes Engine
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: gcr.io/PROJECT_ID/mcp-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: MCP_ENVIRONMENT
          value: production
        - name: MCP_LOG_LEVEL
          value: INFO
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Microsoft Azure

#### Azure Functions
```json
{
  "version": "2.0",
  "functionApp": {
    "id": "/subscriptions/SUBSCRIPTION_ID/resourceGroups/RESOURCE_GROUP/providers/Microsoft.Web/sites/mcp-server"
  },
  "functions": [
    {
      "name": "mcp-handler",
      "config": {
        "bindings": [
          {
            "authLevel": "function",
            "type": "httpTrigger",
            "direction": "in",
            "name": "req",
            "methods": ["get", "post"]
          },
          {
            "type": "http",
            "direction": "out",
            "name": "$return"
          }
        ]
      }
    }
  ]
}
```

#### Azure Container Instances
```yaml
# azure-container.yaml
apiVersion: 2021-07-01
location: West US 2
name: mcp-server
properties:
  containers:
  - name: mcp-server
    properties:
      image: your-registry/mcp-server:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 1.5
      ports:
      - port: 8000
      environmentVariables:
      - name: MCP_ENVIRONMENT
        value: production
      - name: MCP_LOG_LEVEL
        value: INFO
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 8000
```

## Container Deployment

### Docker Setup

#### Basic Dockerfile
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["fastmcp", "run", "src/server.py", "--host", "0.0.0.0", "--port", "8000"]
```

#### Multi-stage Dockerfile for Production
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["fastmcp", "run", "src/server.py", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MCP_ENVIRONMENT=development
      - MCP_LOG_LEVEL=DEBUG
    volumes:
      - .:/app
      - /app/__pycache__
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: mcp_db
      POSTGRES_USER: mcp_user
      POSTGRES_PASSWORD: mcp_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - mcp-server

volumes:
  postgres_data:
```

### Docker Compose for Production
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  mcp-server:
    image: your-registry/mcp-server:latest
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - MCP_ENVIRONMENT=production
      - MCP_LOG_LEVEL=INFO
    secrets:
      - mcp_api_key
    networks:
      - mcp_network
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.prod.conf:/etc/nginx/conf.d/default.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - mcp-server
    networks:
      - mcp_network

networks:
  mcp_network:
    driver: overlay
    attachable: true

secrets:
  mcp_api_key:
    external: true
```

## Serverless Deployment

### AWS Lambda with Serverless Framework
```python
# serverless_handler.py
import json
import asyncio
from fastmcp import FastMCP
from fastmcp.adapters.aws import LambdaAdapter

# Initialize FastMCP
mcp = FastMCP()

@mcp.tool()
async def process_data(data: str) -> dict:
    # Your processing logic
    return {"processed": data.upper()}

# Create Lambda adapter
adapter = LambdaAdapter(mcp)

def lambda_handler(event, context):
    """AWS Lambda handler"""
    try:
        return asyncio.run(adapter.handle(event, context))
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

```yaml
# serverless.yml
service: mcp-server-lambda

provider:
  name: aws
  runtime: python3.11
  stage: ${opt:stage, 'dev'}
  region: us-west-2
  memorySize: 512
  timeout: 30
  environment:
    STAGE: ${self:provider.stage}
    MCP_LOG_LEVEL: ${env:MCP_LOG_LEVEL, 'INFO'}
  iamRoleStatements:
    - Effect: Allow
      Action:
        - logs:CreateLogGroup
        - logs:CreateLogStream
        - logs:PutLogEvents
      Resource: "arn:aws:logs:*:*:*"

functions:
  mcp:
    handler: serverless_handler.lambda_handler
    events:
      - http:
          path: /{proxy+}
          method: ANY
          cors: true
      - http:
          path: /
          method: ANY
          cors: true

plugins:
  - serverless-python-requirements
  - serverless-offline

custom:
  pythonRequirements:
    dockerizePip: true
    slim: true
    strip: false
```

### Google Cloud Functions
```python
# main.py for Google Cloud Functions
import asyncio
from flask import Request
from fastmcp import FastMCP
from fastmcp.adapters.gcp import CloudFunctionAdapter

mcp = FastMCP()

@mcp.tool()
async def hello_world(name: str = "World") -> str:
    return f"Hello, {name}!"

adapter = CloudFunctionAdapter(mcp)

def main(request: Request):
    """Google Cloud Function entry point"""
    return asyncio.run(adapter.handle(request))
```

```yaml
# function.yaml
name: mcp-server
runtime: python311
entryPoint: main
httpsTrigger: {}
environmentVariables:
  MCP_ENVIRONMENT: production
  MCP_LOG_LEVEL: INFO
availableMemoryMb: 512
timeout: 60s
```

### Azure Functions
```python
# function_app.py
import azure.functions as func
import asyncio
from fastmcp import FastMCP
from fastmcp.adapters.azure import AzureFunctionAdapter

mcp = FastMCP()

@mcp.tool()
async def azure_function(message: str) -> str:
    return f"Azure processed: {message}"

adapter = AzureFunctionAdapter(mcp)

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="mcp", methods=["GET", "POST"])
def mcp_handler(req: func.HttpRequest) -> func.HttpResponse:
    return asyncio.run(adapter.handle(req))
```

## Traditional Server Deployment

### systemd Service
```ini
# /etc/systemd/system/mcp-server.service
[Unit]
Description=MCP Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=mcp
Group=mcp
WorkingDirectory=/opt/mcp-server
Environment=PATH=/opt/mcp-server/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=MCP_ENVIRONMENT=production
Environment=MCP_LOG_LEVEL=INFO
ExecStart=/opt/mcp-server/venv/bin/fastmcp run src/server.py --host 0.0.0.0 --port 8000
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StartLimitInterval=60s
StartLimitBurst=3

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/mcp-server/logs
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy
```nginx
# /etc/nginx/sites-available/mcp-server
upstream mcp_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    listen [::]:80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name your-domain.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://mcp_backend;
        access_log off;
    }

    # Static files (if any)
    location /static/ {
        alias /opt/mcp-server/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

### Process Management with Supervisor
```ini
# /etc/supervisor/conf.d/mcp-server.conf
[program:mcp-server]
command=/opt/mcp-server/venv/bin/fastmcp run src/server.py --host 127.0.0.1 --port 8000
directory=/opt/mcp-server
user=mcp
group=mcp
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/supervisor/mcp-server.log
stderr_logfile=/var/log/supervisor/mcp-server.log
environment=MCP_ENVIRONMENT=production,MCP_LOG_LEVEL=INFO

[program:mcp-server-worker-1]
command=/opt/mcp-server/venv/bin/fastmcp run src/server.py --host 127.0.0.1 --port 8001
directory=/opt/mcp-server
user=mcp
group=mcp
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/supervisor/mcp-server-worker-1.log
stderr_logfile=/var/log/supervisor/mcp-server-worker-1.log
environment=MCP_ENVIRONMENT=production,MCP_LOG_LEVEL=INFO

[program:mcp-server-worker-2]
command=/opt/mcp-server/venv/bin/fastmcp run src/server.py --host 127.0.0.1 --port 8002
directory=/opt/mcp-server
user=mcp
group=mcp
autostart=true
autorestart=true
startretries=3
stdout_logfile=/var/log/supervisor/mcp-server-worker-2.log
stderr_logfile=/var/log/supervisor/mcp-server-worker-2.log
environment=MCP_ENVIRONMENT=production,MCP_LOG_LEVEL=INFO

[group:mcp-servers]
programs=mcp-server,mcp-server-worker-1,mcp-server-worker-2
priority=999
```

## CI/CD Integration

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy MCP Server

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install fastmcp[dev]
    
    - name: Run tests
      run: |
        fastmcp test --coverage
        fastmcp validate src/server.py
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=sha
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to AWS ECS
      env:
        AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        AWS_REGION: us-west-2
      run: |
        aws ecs update-service \
          --cluster mcp-cluster \
          --service mcp-server \
          --force-new-deployment
    
    - name: Verify deployment
      run: |
        sleep 60  # Wait for deployment
        curl -f https://api.yourdomain.com/health || exit 1
```

### GitLab CI
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE
  DOCKER_TAG: $CI_COMMIT_SHA

test:
  stage: test
  image: python:3.11
  services:
    - redis:7-alpine
    - postgres:15-alpine
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_password
  before_script:
    - pip install -r requirements.txt
    - pip install fastmcp[dev]
  script:
    - fastmcp test --coverage
    - fastmcp validate src/server.py
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_IMAGE:$DOCKER_TAG .
    - docker push $DOCKER_IMAGE:$DOCKER_TAG
    - docker tag $DOCKER_IMAGE:$DOCKER_TAG $DOCKER_IMAGE:latest
    - docker push $DOCKER_IMAGE:latest
  only:
    - main

deploy-staging:
  stage: deploy
  image: alpine/helm:latest
  environment:
    name: staging
    url: https://staging.yourdomain.com
  script:
    - helm upgrade --install mcp-server-staging ./helm-chart \
        --set image.tag=$DOCKER_TAG \
        --set environment=staging \
        --namespace staging
  only:
    - main

deploy-production:
  stage: deploy
  image: alpine/helm:latest
  environment:
    name: production
    url: https://yourdomain.com
  when: manual
  script:
    - helm upgrade --install mcp-server-prod ./helm-chart \
        --set image.tag=$DOCKER_TAG \
        --set environment=production \
        --namespace production
  only:
    - main
```

### Jenkins Pipeline
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = "your-registry/mcp-server"
        DOCKER_TAG = "${env.BUILD_NUMBER}"
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/your-org/mcp-server.git'
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pip install fastmcp[dev]
                    fastmcp test --coverage
                    fastmcp validate src/server.py
                '''
            }
            post {
                always {
                    publishCoverage adapters: [
                        coberturaAdapter('coverage.xml')
                    ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                }
            }
        }
        
        stage('Build') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }
        
        stage('Push') {
            steps {
                script {
                    docker.withRegistry('https://your-registry', 'docker-registry-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push("latest")
                    }
                }
            }
        }
        
        stage('Deploy to Staging') {
            steps {
                sh """
                    kubectl set image deployment/mcp-server mcp-server=${DOCKER_IMAGE}:${DOCKER_TAG} -n staging
                    kubectl rollout status deployment/mcp-server -n staging
                """
            }
        }
        
        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            input {
                message "Deploy to production?"
                ok "Deploy"
            }
            steps {
                sh """
                    kubectl set image deployment/mcp-server mcp-server=${DOCKER_IMAGE}:${DOCKER_TAG} -n production
                    kubectl rollout status deployment/mcp-server -n production
                """
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        success {
            slackSend(
                channel: '#deployments',
                color: 'good',
                message: "✅ MCP Server deployed successfully - Build #${env.BUILD_NUMBER}"
            )
        }
        failure {
            slackSend(
                channel: '#deployments',
                color: 'danger',
                message: "❌ MCP Server deployment failed - Build #${env.BUILD_NUMBER}"
            )
        }
    }
}
```

## Environment Configuration

### Environment Variables
```bash
# Production environment variables
export MCP_ENVIRONMENT=production
export MCP_LOG_LEVEL=INFO
export MCP_SECRET_KEY=your-secret-key
export MCP_DATABASE_URL=postgresql://user:pass@localhost/db
export MCP_REDIS_URL=redis://localhost:6379/0
export MCP_API_KEY=your-api-key

# Performance settings
export MCP_WORKERS=4
export MCP_WORKER_CONNECTIONS=1000
export MCP_KEEPALIVE_TIMEOUT=2
export MCP_MAX_REQUESTS=1000

# Security settings
export MCP_ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
export MCP_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
export MCP_RATE_LIMIT=100/minute

# Monitoring
export MCP_SENTRY_DSN=your-sentry-dsn
export MCP_DATADOG_API_KEY=your-datadog-key
export MCP_NEW_RELIC_LICENSE_KEY=your-newrelic-key
```

### Configuration Files

#### Production Config
```json
{
  "name": "mcp-server-prod",
  "version": "1.0.0",
  "environment": "production",
  
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "keepalive_timeout": 2,
    "max_requests": 1000,
    "max_requests_jitter": 100
  },
  
  "logging": {
    "level": "INFO",
    "format": "json",
    "handlers": {
      "file": {
        "filename": "/var/log/mcp-server/server.log",
        "max_bytes": 10485760,
        "backup_count": 5
      },
      "syslog": {
        "address": ["localhost", 514],
        "facility": "local0"
      }
    }
  },
  
  "database": {
    "url": "${DATABASE_URL}",
    "pool_size": 20,
    "max_overflow": 0,
    "pool_timeout": 30,
    "pool_recycle": 3600
  },
  
  "cache": {
    "backend": "redis",
    "url": "${REDIS_URL}",
    "default_timeout": 300,
    "key_prefix": "mcp:",
    "connection_pool": {
      "max_connections": 50
    }
  },
  
  "security": {
    "secret_key": "${SECRET_KEY}",
    "allowed_hosts": ["yourdomain.com", "api.yourdomain.com"],
    "cors": {
      "allowed_origins": ["https://yourdomain.com"],
      "allowed_methods": ["GET", "POST", "PUT", "DELETE"],
      "allowed_headers": ["*"],
      "allow_credentials": true
    },
    "rate_limiting": {
      "default": "100/minute",
      "per_endpoint": {
        "/api/heavy-operation": "10/minute"
      }
    }
  },
  
  "monitoring": {
    "metrics": {
      "enabled": true,
      "prometheus": {
        "port": 9090,
        "path": "/metrics"
      }
    },
    "tracing": {
      "enabled": true,
      "jaeger": {
        "agent_host": "localhost",
        "agent_port": 6831
      }
    },
    "health_checks": {
      "enabled": true,
      "path": "/health",
      "detailed_path": "/health/detailed"
    }
  }
}
```

## Monitoring and Observability

### Health Checks
```python
# health_checks.py
from fastmcp import FastMCP
from fastmcp.monitoring import HealthCheck
import asyncio
import aioredis
import asyncpg

mcp = FastMCP()

@mcp.health_check("database")
async def check_database():
    """Check database connectivity"""
    try:
        conn = await asyncpg.connect("postgresql://user:pass@localhost/db")
        await conn.execute("SELECT 1")
        await conn.close()
        return {"status": "healthy", "response_time": "< 100ms"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@mcp.health_check("redis")
async def check_redis():
    """Check Redis connectivity"""
    try:
        redis = aioredis.from_url("redis://localhost")
        await redis.ping()
        await redis.close()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@mcp.health_check("external_api")
async def check_external_api():
    """Check external API dependency"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.external.com/health", timeout=5) as response:
                if response.status == 200:
                    return {"status": "healthy"}
                else:
                    return {"status": "degraded", "status_code": response.status}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Prometheus Metrics
```python
# metrics.py
from fastmcp import FastMCP
from fastmcp.monitoring import prometheus_metrics
from prometheus_client import Counter, Histogram, Gauge
import time

# Custom metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Active connections')

mcp = FastMCP()

@mcp.middleware()
async def metrics_middleware(request, next):
    """Collect request metrics"""
    start_time = time.time()
    method = request.get('method', 'unknown')
    endpoint = request.get('path', 'unknown')
    
    try:
        response = await next(request)
        status = 'success'
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        return response
    except Exception as e:
        status = 'error'
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        raise
    finally:
        duration = time.time() - start_time
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

# Enable built-in metrics
mcp.enable_prometheus_metrics(port=9090, path="/metrics")
```

### Structured Logging
```python
# logging_config.py
import structlog
from fastmcp import FastMCP
from fastmcp.logging import setup_logging

# Configure structured logging
setup_logging(
    level="INFO",
    format="json",
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ]
)

mcp = FastMCP()

# Add request logging middleware
@mcp.middleware()
async def logging_middleware(request, next):
    """Log all requests with structured data"""
    logger = structlog.get_logger()
    
    request_id = request.get('request_id', 'unknown')
    method = request.get('method', 'unknown')
    path = request.get('path', 'unknown')
    user_id = request.get('user', {}).get('id', 'anonymous')
    
    logger.info(
        "Request started",
        request_id=request_id,
        method=method,
        path=path,
        user_id=user_id
    )
    
    try:
        response = await next(request)
        logger.info(
            "Request completed",
            request_id=request_id,
            status="success"
        )
        return response
    except Exception as e:
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        raise
```

### Distributed Tracing
```python
# tracing.py
from fastmcp import FastMCP
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument libraries
AioHttpClientInstrumentor().instrument()
AsyncPGInstrumentor().instrument()

mcp = FastMCP()

@mcp.middleware()
async def tracing_middleware(request, next):
    """Add distributed tracing to requests"""
    with tracer.start_as_current_span("mcp_request") as span:
        span.set_attribute("http.method", request.get('method', 'unknown'))
        span.set_attribute("http.url", request.get('path', 'unknown'))
        span.set_attribute("user.id", request.get('user', {}).get('id', 'anonymous'))
        
        try:
            response = await next(request)
            span.set_attribute("http.status_code", 200)
            return response
        except Exception as e:
            span.set_attribute("http.status_code", 500)
            span.record_exception(e)
            raise
```

## Security Considerations

### SSL/TLS Configuration
```nginx
# Enhanced SSL configuration
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    
    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'";
}
```

### Secrets Management
```python
# secrets_manager.py
import os
import boto3
from google.cloud import secretmanager
from azure.keyvault.secrets import SecretClient
from fastmcp.config import Config

class SecretsManager:
    """Centralized secrets management"""
    
    def __init__(self, provider: str = "aws"):
        self.provider = provider
        self._client = self._get_client()
    
    def _get_client(self):
        if self.provider == "aws":
            return boto3.client('secretsmanager')
        elif self.provider == "gcp":
            return secretmanager.SecretManagerServiceClient()
        elif self.provider == "azure":
            from azure.identity import DefaultAzureCredential
            credential = DefaultAzureCredential()
            return SecretClient(
                vault_url=os.getenv("AZURE_KEY_VAULT_URL"), 
                credential=credential
            )
    
    async def get_secret(self, secret_name: str) -> str:
        """Get secret from provider"""
        if self.provider == "aws":
            response = self._client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
        elif self.provider == "gcp":
            name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_name}/versions/latest"
            response = self._client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        elif self.provider == "azure":
            secret = self._client.get_secret(secret_name)
            return secret.value

# Usage in FastMCP configuration
secrets = SecretsManager(provider=os.getenv("SECRETS_PROVIDER", "aws"))

async def get_config():
    return Config(
        database_url=await secrets.get_secret("database-url"),
        api_key=await secrets.get_secret("api-key"),
        secret_key=await secrets.get_secret("secret-key")
    )
```

### Security Hardening
```python
# security.py
from fastmcp import FastMCP
from fastmcp.security import SecurityMiddleware, RateLimiter
import hashlib
import hmac
import time

mcp = FastMCP()

# Rate limiting
rate_limiter = RateLimiter(
    default_limit="100/minute",
    storage="redis://localhost:6379"
)

@mcp.middleware()
async def security_middleware(request, next):
    """Comprehensive security middleware"""
    
    # Request validation
    if not _validate_request_size(request):
        raise ValueError("Request too large")
    
    # IP filtering
    if not _is_allowed_ip(request.get('remote_addr')):
        raise PermissionError("IP not allowed")
    
    # Rate limiting
    if not await rate_limiter.allow_request(request):
        raise PermissionError("Rate limit exceeded")
    
    # HMAC signature verification
    if not _verify_signature(request):
        raise PermissionError("Invalid signature")
    
    return await next(request)

def _validate_request_size(request):
    """Validate request size"""
    max_size = 1024 * 1024  # 1MB
    content_length = int(request.get('content-length', 0))
    return content_length <= max_size

def _is_allowed_ip(remote_addr):
    """Check if IP is in allowed list"""
    allowed_ips = os.getenv('ALLOWED_IPS', '').split(',')
    if not allowed_ips or allowed_ips == ['']:
        return True
    return remote_addr in allowed_ips

def _verify_signature(request):
    """Verify HMAC signature"""
    secret = os.getenv('WEBHOOK_SECRET')
    if not secret:
        return True
    
    signature = request.get('headers', {}).get('X-Signature')
    if not signature:
        return False
    
    body = request.get('body', '')
    expected = hmac.new(
        secret.encode(),
        body.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, f"sha256={expected}")
```

## Scaling and Load Balancing

### Horizontal Scaling
```yaml
# kubernetes-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 3
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
```

### Load Balancer Configuration
```yaml
# aws-load-balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-lb
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout: "60"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: "tcp"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-protocol: "http"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-path: "/health"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-interval: "10"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-healthy-threshold: "2"
    service.beta.kubernetes.io/aws-load-balancer-healthcheck-unhealthy-threshold: "3"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
  selector:
    app: mcp-server
```

### Database Connection Pooling
```python
# database.py
from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
import os

# Production database configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# Connection pool configuration
engine = create_async_engine(
    DATABASE_URL,
    # Connection pool settings
    pool_size=20,                    # Number of connections to maintain
    max_overflow=0,                  # Additional connections beyond pool_size
    pool_timeout=30,                 # Timeout to get connection from pool
    pool_recycle=3600,              # Recycle connections after 1 hour
    pool_pre_ping=True,             # Verify connections before use
    # Performance settings
    echo=False,                     # Don't log SQL in production
    future=True,
    # Connection arguments
    connect_args={
        "command_timeout": 60,
        "server_settings": {
            "application_name": "mcp-server",
            "jit": "off",           # Disable JIT for consistent performance
        },
    }
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession
)

mcp = FastMCP()

@mcp.middleware()
async def database_middleware(request, next):
    """Provide database session for each request"""
    async with AsyncSessionLocal() as session:
        request["db"] = session
        try:
            response = await next(request)
            await session.commit()
            return response
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

## Troubleshooting

### Common Deployment Issues

#### Port Conflicts
```bash
# Check if port is in use
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000

# Find and kill process using port
sudo fuser -k 8000/tcp

# Use alternative port
fastmcp run src/server.py --port 8001
```

#### Permission Errors
```bash
# Fix file permissions
sudo chown -R mcp:mcp /opt/mcp-server
sudo chmod -R 755 /opt/mcp-server
sudo chmod +x /opt/mcp-server/venv/bin/fastmcp

# Fix log directory permissions
sudo mkdir -p /var/log/mcp-server
sudo chown mcp:mcp /var/log/mcp-server
sudo chmod 755 /var/log/mcp-server
```

#### Memory Issues
```bash
# Monitor memory usage
htop
free -h
ps aux --sort=-%mem | head -10

# Adjust worker count based on available memory
# Rule of thumb: (RAM - OS - Other processes) / Worker memory usage
fastmcp run src/server.py --workers 2  # Reduce workers if low memory
```

#### Database Connection Issues
```python
# Test database connectivity
import asyncio
import asyncpg

async def test_db():
    try:
        conn = await asyncpg.connect("postgresql://user:pass@host:5432/db")
        result = await conn.fetchval("SELECT version()")
        print(f"Connected successfully: {result}")
        await conn.close()
    except Exception as e:
        print(f"Connection failed: {e}")

asyncio.run(test_db())
```

### Debugging Production Issues

#### Enable Debug Logging
```python
# Temporarily enable debug logging
import logging
import structlog

logging.basicConfig(level=logging.DEBUG)
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=False)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Or via environment variable
export FASTMCP_LOG_LEVEL=DEBUG
fastmcp run src/server.py
```

#### Performance Profiling
```python
# profile.py
import cProfile
import pstats
from fastmcp import FastMCP

mcp = FastMCP()

def profile_app():
    """Profile application performance"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your test workload here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions

if __name__ == "__main__":
    profile_app()
```

#### Memory Profiling
```python
# memory_profile.py
from memory_profiler import profile
from fastmcp import FastMCP

mcp = FastMCP()

@profile
def memory_intensive_function():
    """Function to profile memory usage"""
    # Your memory-intensive code here
    pass

# Run with: python -m memory_profiler memory_profile.py
```

### Health Check Scripts
```bash
#!/bin/bash
# health_check.sh

URL="http://localhost:8000/health"
TIMEOUT=10
RETRY_COUNT=3

for i in $(seq 1 $RETRY_COUNT); do
    if curl -f --connect-timeout $TIMEOUT "$URL" > /dev/null 2>&1; then
        echo "Health check passed"
        exit 0
    fi
    echo "Health check failed, retry $i/$RETRY_COUNT"
    sleep 2
done

echo "Health check failed after $RETRY_COUNT attempts"
exit 1
```

### Log Analysis
```bash
# Analyze logs for errors
grep -E "(ERROR|CRITICAL)" /var/log/mcp-server/server.log | tail -20

# Check for performance issues
grep -E "(slow|timeout|502|503|504)" /var/log/nginx/access.log | tail -20

# Monitor real-time logs
tail -f /var/log/mcp-server/server.log | grep ERROR

# Analyze response times
awk '$9 ~ /^[45]/ {print $0}' /var/log/nginx/access.log | tail -20
```

### Recovery Procedures
```bash
#!/bin/bash
# recovery.sh - Automated recovery script

echo "Starting recovery procedure..."

# Check if service is running
if ! systemctl is-active --quiet mcp-server; then
    echo "Service not running, starting..."
    systemctl start mcp-server
    sleep 10
fi

# Check health endpoint
if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "Health check failed, restarting service..."
    systemctl restart mcp-server
    sleep 20
fi

# Check database connectivity
if ! python3 -c "import asyncio; import asyncpg; asyncio.run(asyncpg.connect('$DATABASE_URL').close())" > /dev/null 2>&1; then
    echo "Database connection failed, check database status"
    # Add database recovery logic here
fi

# Check disk space
DISK_USAGE=$(df /opt/mcp-server | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "Disk usage critical: ${DISK_USAGE}%"
    # Add cleanup logic here
    find /opt/mcp-server/logs -name "*.log" -mtime +7 -delete
fi

# Verify final health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "Recovery successful"
    exit 0
else
    echo "Recovery failed, manual intervention required"
    exit 1
fi
```

## Summary

FastMCP deployment supports multiple strategies:

1. **Development**: Local development servers with auto-reload
2. **Serverless**: AWS Lambda, Google Cloud Functions, Azure Functions
3. **Containers**: Docker, Kubernetes, container orchestration
4. **Traditional**: VMs, bare metal with process managers
5. **Cloud Native**: Managed services, auto-scaling, load balancing

Key deployment considerations:
- **Environment Configuration**: Proper secret management and environment variables
- **Security**: SSL/TLS, authentication, authorization, and hardening
- **Monitoring**: Health checks, metrics, logging, and distributed tracing
- **Scaling**: Horizontal scaling, load balancing, and connection pooling
- **CI/CD**: Automated testing, building, and deployment pipelines
- **Recovery**: Health checks, monitoring, and automated recovery procedures

Choose the deployment strategy that best fits your infrastructure, scaling requirements, and operational expertise.