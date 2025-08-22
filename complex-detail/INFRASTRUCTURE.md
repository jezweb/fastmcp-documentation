# FastMCP Infrastructure Deployment Guide

> **Note**: These are advanced deployment patterns for enterprise environments. Most users should use [FastMCP Cloud](https://fastmcp.cloud) for simpler deployment.

## Cloud Provider Deployments

### AWS Deployment Options

#### AWS Lambda
```python
# lambda_handler.py
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("lambda-server")

@mcp.tool()
async def process_event(event: dict):
    """Process Lambda event."""
    return {"processed": event}

def lambda_handler(event, context):
    """AWS Lambda handler."""
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        mcp.handle_lambda_event(event)
    )
    return result
```

**SAM Template:**
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  MCPFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: .
      Handler: lambda_handler.lambda_handler
      Runtime: python3.11
      Timeout: 300
      MemorySize: 512
      Environment:
        Variables:
          API_KEY: !Ref ApiKey
```

#### AWS ECS/Fargate
```yaml
# task-definition.json
{
  "family": "mcp-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [{
    "name": "mcp-server",
    "image": "your-ecr-repo/mcp-server:latest",
    "portMappings": [{
      "containerPort": 5000,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "API_KEY", "value": "your-key"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/mcp-server",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

#### AWS EC2 User Data Script
```bash
#!/bin/bash
yum update -y
yum install -y python3 git

# Clone repository
git clone https://github.com/your-org/mcp-server.git /opt/mcp-server
cd /opt/mcp-server

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
ExecStart=/usr/bin/python3 -m fastmcp run server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

systemctl enable mcp-server
systemctl start mcp-server
```

### Azure Deployment

#### Azure Functions
```python
# function_app.py
import azure.functions as func
from fastmcp import FastMCP
import asyncio

mcp = FastMCP("azure-function")

@mcp.tool()
async def process_request(data: dict):
    return {"processed": data}

async def main(req: func.HttpRequest) -> func.HttpResponse:
    result = await mcp.handle_azure_request(req)
    return func.HttpResponse(result, mimetype="application/json")
```

#### Azure Container Instances
```yaml
# aci-deploy.yaml
apiVersion: 2019-12-01
location: eastus
name: mcp-server-group
properties:
  containers:
  - name: mcp-server
    properties:
      image: mcpserver.azurecr.io/mcp-server:latest
      resources:
        requests:
          cpu: 1
          memoryInGb: 1.5
      ports:
      - port: 5000
      environmentVariables:
      - name: API_KEY
        secureValue: your-key
  osType: Linux
  ipAddress:
    type: Public
    ports:
    - protocol: tcp
      port: 5000
```

### Google Cloud Platform

#### Cloud Run
```dockerfile
# Dockerfile for Cloud Run
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run requires PORT env var
ENV PORT 8080
EXPOSE 8080

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 server:app
```

**Deploy Command:**
```bash
gcloud run deploy mcp-server \
  --image gcr.io/PROJECT-ID/mcp-server \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars API_KEY=your-key
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
        image: gcr.io/PROJECT-ID/mcp-server:latest
        ports:
        - containerPort: 5000
        env:
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server-service
spec:
  selector:
    app: mcp-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 5000
  type: LoadBalancer
```

## Kubernetes Deployment

### Helm Chart Structure
```
mcp-server-chart/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── ingress.yaml
```

**values.yaml:**
```yaml
replicaCount: 2

image:
  repository: your-registry/mcp-server
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  className: nginx
  hosts:
    - host: mcp.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 250m
    memory: 256Mi

env:
  - name: LOG_LEVEL
    value: info
  
secrets:
  - name: API_KEY
    value: your-encrypted-key
```

## Terraform Infrastructure as Code

### AWS Infrastructure
```hcl
# main.tf
provider "aws" {
  region = "us-east-1"
}

resource "aws_ecs_cluster" "mcp_cluster" {
  name = "mcp-cluster"
}

resource "aws_ecs_service" "mcp_service" {
  name            = "mcp-service"
  cluster         = aws_ecs_cluster.mcp_cluster.id
  task_definition = aws_ecs_task_definition.mcp_task.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.mcp_sg.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.mcp_tg.arn
    container_name   = "mcp-server"
    container_port   = 5000
  }
}

resource "aws_ecs_task_definition" "mcp_task" {
  family                   = "mcp-task"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = "512"
  memory                  = "1024"

  container_definitions = jsonencode([{
    name  = "mcp-server"
    image = "${aws_ecr_repository.mcp_repo.repository_url}:latest"
    portMappings = [{
      containerPort = 5000
    }]
    environment = [
      {
        name  = "API_KEY"
        value = var.api_key
      }
    ]
  }])
}
```

## CI/CD Pipelines

### GitHub Actions
```yaml
name: Deploy Infrastructure

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Build and push Docker image
        run: |
          docker build -t mcp-server .
          docker tag mcp-server:latest $ECR_REGISTRY/mcp-server:latest
          docker push $ECR_REGISTRY/mcp-server:latest
      
      - name: Deploy to ECS
        run: |
          aws ecs update-service \
            --cluster mcp-cluster \
            --service mcp-service \
            --force-new-deployment
```

### GitLab CI/CD
```yaml
# .gitlab-ci.yml
stages:
  - build
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy:
  stage: deploy
  script:
    - kubectl set image deployment/mcp-server mcp-server=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main
```

## Load Balancing & Scaling

### Auto-scaling Configuration
```yaml
# AWS Auto-scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mcp-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mcp-server
  minReplicas: 2
  maxReplicas: 10
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
```

## Monitoring & Observability

### Prometheus & Grafana
```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['mcp-server:5000']
    metrics_path: '/metrics'
```

### ELK Stack
```yaml
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_kubernetes_metadata:
        host: ${NODE_NAME}
        matchers:
        - logs_path:
            logs_path: "/var/lib/docker/containers/"

output.elasticsearch:
  hosts: ['elasticsearch:9200']
```

## Disaster Recovery

### Backup Strategy
```bash
#!/bin/bash
# backup-infrastructure.sh

# Backup Docker volumes
docker run --rm -v mcp-data:/data -v $(pwd):/backup ubuntu tar czf /backup/data-$(date +%Y%m%d).tar.gz /data

# Backup Kubernetes resources
kubectl get all --all-namespaces -o yaml > k8s-backup-$(date +%Y%m%d).yaml

# Backup database
pg_dump $DATABASE_URL | gzip > db-backup-$(date +%Y%m%d).sql.gz

# Upload to S3
aws s3 cp . s3://backup-bucket/$(date +%Y%m%d)/ --recursive --exclude "*" --include "*.gz" --include "*.yaml"
```

### Recovery Plan
1. **RTO (Recovery Time Objective)**: < 1 hour
2. **RPO (Recovery Point Objective)**: < 24 hours
3. **Backup retention**: 30 days
4. **Test recovery**: Monthly

## Security Best Practices

### Network Security
```yaml
# NetworkPolicy for Kubernetes
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mcp-server-netpol
spec:
  podSelector:
    matchLabels:
      app: mcp-server
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx
    ports:
    - protocol: TCP
      port: 5000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

### Secrets Management
```bash
# Using HashiCorp Vault
vault kv put secret/mcp-server \
  api_key="$API_KEY" \
  db_password="$DB_PASSWORD"

# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name mcp-server-secrets \
  --secret-string '{"api_key":"value","db_password":"value"}'
```

## Cost Optimization

### Resource Right-sizing
- Monitor actual resource usage
- Use spot instances for non-critical workloads
- Implement auto-scaling based on demand
- Use reserved instances for predictable workloads

### Cost Monitoring
```python
# cost_monitor.py
import boto3

ce = boto3.client('ce')

def get_monthly_cost():
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': '2024-01-01',
            'End': '2024-01-31'
        },
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
        Filter={
            'Tags': {
                'Key': 'Service',
                'Values': ['mcp-server']
            }
        }
    )
    return response
```

## Migration to FastMCP Cloud

When infrastructure becomes too complex:

1. **Evaluate costs**: Compare infrastructure costs vs FastMCP Cloud
2. **Simplify architecture**: Remove unnecessary complexity
3. **Prepare for migration**: Ensure code is cloud-ready
4. **Test on cloud**: Deploy test instance to FastMCP Cloud
5. **Migrate gradually**: Move non-critical services first
6. **Monitor and optimize**: Use FastMCP Cloud monitoring tools

## Additional Resources

- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Azure Architecture Center](https://docs.microsoft.com/azure/architecture/)
- [Google Cloud Architecture Guide](https://cloud.google.com/architecture)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform Documentation](https://www.terraform.io/docs/)