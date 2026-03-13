# Water Meter AI Recognition - Docker Deployment

## 🐳 Quick Start with Docker

### Prerequisites
- Docker Desktop installed (https://www.docker.com/get-started)
- At least 4GB RAM available for Docker
- Models placed in `model/` directory

### 🚀 Quick Start (Windows)

```cmd
# 1. Build the Docker image
docker-build.bat build

# 2. Run the container
docker-build.bat run

# 3. Check status
docker-build.bat status
```

### 🚀 Quick Start (Linux/Mac)

```bash
# Make script executable
chmod +x docker-build.sh

# Build and run
./docker-build.sh build
./docker-build.sh run
```

### 📋 Available Commands

| Command | Description |
|---------|-------------|
| `build` | Build Docker image from Dockerfile |
| `run` | Start container in detached mode |
| `stop` | Stop and remove container |
| `restart` | Restart running container |
| `logs` | View container logs (live) |
| `status` | Show container status |
| `test` | Test API health endpoint |
| `clean` | Remove containers and cleanup resources |

### 🔗 API Endpoints

Once container is running:

- **API Root**: http://localhost:8000
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 🧪 Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Predict (example)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_encoded_image_string"}'
```

### 📁 Project Structure

```
Project/
├── api/                    # FastAPI application
│   ├── main.py            # API server
│   └── requirements.txt   # Python dependencies
├── src/                    # Source code (M1-M4 modules)
├── model/                  # ML model weights (.pth files)
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker orchestration
├── .dockerignore          # Files to exclude from image
└── docker-build.bat       # Windows build script
```

### ⚙️ Configuration

#### Memory & CPU Limits

Edit `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'      # Adjust based on your CPU
      memory: 4G     # Adjust based on your RAM
```

#### Port Mapping

To change from port 8000:

```yaml
ports:
  - "8080:8000"  # Maps localhost:8080 to container:8000
```

### 🔍 Troubleshooting

#### Container won't start

```bash
# Check logs
docker-build.bat logs

# Check container status
docker-compose ps

# Rebuild without cache
docker-compose build --no-cache
```

#### Port already in use

```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process or change port in docker-compose.yml
```

#### Out of memory

Increase memory limit in Docker Desktop:
1. Docker Desktop → Settings → Resources
2. Increase "Memory" to at least 4GB
3. Click "Apply & Restart"

#### API returns 500 errors

```bash
# Check container logs for errors
docker-compose logs -f water-meter-api

# Common issues:
# - Missing model files in model/
# - Incorrect model paths in code
# - Missing dependencies in requirements.txt
```

### 📊 Monitoring

#### View Logs

```bash
# Live logs
docker-compose logs -f water-meter-api

# Last 100 lines
docker-compose logs --tail=100 water-meter-api
```

#### Resource Usage

```bash
# Container stats
docker stats water-meter-api
```

### 🔄 Updating Models

Without rebuilding the image:

```bash
# 1. Place new models in model/ directory
# 2. Restart container
docker-compose restart water-meter-api
```

The `model/` directory is mounted as a volume, so changes are reflected immediately.

### 🧹 Cleanup

```bash
# Stop and remove containers
docker-build.bat stop

# Remove images and volumes
docker-build.bat clean
```

### 🌐 Production Deployment

#### Using Nginx Reverse Proxy

Uncomment the `nginx` service in `docker-compose.yml` and create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server water-meter-api:8000;
    }

    server {
        listen 80;
        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

#### Environment Variables

```yaml
environment:
  - LOG_LEVEL=info
  - MODEL_PATH=/app/model
  - CUDA_VISIBLE_DEVICES=""  # Disable GPU in Docker
```

### 📝 Notes

- **OpenCV**: Uses `opencv-python-headless` for server environments
- **GPU Support**: Requires nvidia-docker runtime (not configured by default)
- **Persistence**: Logs and data are mounted as volumes
- **Health Check**: Automatic health monitoring every 30s

### 🔗 Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

---

**Need help?** Check the logs: `docker-build.bat logs`
