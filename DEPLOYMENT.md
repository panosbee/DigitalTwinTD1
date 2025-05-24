# 🚀 Digital Twin T1D SDK - Deployment Guide

## Quick Start with Docker

### 1. Single Container Deployment

```bash
# Build the image
docker build -t digitaltwin-t1d .

# Run the container
docker run -p 8080:8080 digitaltwin-t1d
```

Your API is now available at `http://localhost:8080`

### 2. Full Stack Deployment with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

This will start:
- **API Server**: http://localhost:8080
- **Dashboard**: http://localhost:8081
- **Jupyter Lab**: http://localhost:8888
- **Redis Cache**: localhost:6379

## 🌐 REST API Endpoints

### Health Check
```bash
curl http://localhost:8080/health
```

### Connect Device
```bash
curl -X POST http://localhost:8080/device/connect \
  -H "Content-Type: application/json" \
  -d '{"device_type": "dexcom_g6"}'
```

### Predict Glucose
```bash
curl -X POST http://localhost:8080/predict/glucose \
  -H "Content-Type: application/json" \
  -d '{"horizon_minutes": 30}'
```

### Get Recommendations
```bash
curl -X POST http://localhost:8080/recommendations \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Interactive API Documentation
Visit `http://localhost:8080/docs` for interactive Swagger UI

## ☁️ Cloud Deployment

### AWS
```bash
# Using ECS
aws ecs create-cluster --cluster-name digitaltwin-cluster
aws ecs register-task-definition --cli-input-json file://ecs-task.json
aws ecs create-service --cluster digitaltwin-cluster --service-name dt-api --task-definition digitaltwin-t1d:1
```

### Google Cloud
```bash
# Using Cloud Run
gcloud run deploy digitaltwin-api \
  --image gcr.io/PROJECT-ID/digitaltwin-t1d \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure
```bash
# Using Container Instances
az container create \
  --resource-group myResourceGroup \
  --name digitaltwin-api \
  --image digitaltwin/t1d-sdk:latest \
  --dns-name-label digitaltwin-demo \
  --ports 8080
```

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration
SDK_MODE=production         # production, research, mobile, clinical
API_HOST=0.0.0.0
API_PORT=8080

# Redis Configuration  
REDIS_URL=redis://localhost:6379

# Security
API_KEY=your-secret-key
ENABLE_AUTH=true
```

### Production Checklist
- [ ] Set secure API keys
- [ ] Enable HTTPS/SSL
- [ ] Configure CORS properly
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure auto-scaling
- [ ] Set up backup strategy
- [ ] Enable logging to cloud service

## 📊 Monitoring

### Prometheus Metrics
The API exposes metrics at `/metrics`:
- Request count
- Response times
- Error rates
- Active connections

### Health Checks
- `/health` - Basic health check
- `/health/deep` - Deep health check including dependencies

## 🐛 Troubleshooting

### Container won't start
```bash
# Check logs
docker logs digitaltwin-api

# Check resource usage
docker stats
```

### API not responding
```bash
# Test connectivity
curl http://localhost:8080/health

# Check if port is open
netstat -an | grep 8080
```

### Redis connection issues
```bash
# Test Redis
redis-cli ping

# Check Redis logs
docker logs dt-redis
```

## 📚 Additional Resources

- API Documentation: http://localhost:8080/docs
- Redoc Documentation: http://localhost:8080/redoc
- GitHub Issues: https://github.com/digitaltwin-t1d/sdk/issues
- Discord Community: [Join Here]

---

