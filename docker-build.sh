#!/bin/bash
# Water Meter AI - Docker Build & Run Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Water Meter AI - Docker Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://www.docker.com/get-started"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: docker-compose not found, using 'docker compose' instead${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Menu
case "${1:-}" in
  build)
    echo -e "${YELLOW}Building Docker image...${NC}"
    $DOCKER_COMPOSE build --no-cache
    echo -e "${GREEN}✓ Build completed${NC}"
    ;;

  run)
    echo -e "${YELLOW}Starting Docker container...${NC}"
    $DOCKER_COMPOSE up -d
    echo -e "${GREEN}✓ Container started${NC}"
    echo -e "API available at: ${GREEN}http://localhost:8000${NC}"
    echo -e "API docs at: ${GREEN}http://localhost:8000/docs${NC}"
    ;;

  stop)
    echo -e "${YELLOW}Stopping Docker container...${NC}"
    $DOCKER_COMPOSE down
    echo -e "${GREEN}✓ Container stopped${NC}"
    ;;

  restart)
    echo -e "${YELLOW}Restarting Docker container...${NC}"
    $DOCKER_COMPOSE restart
    echo -e "${GREEN}✓ Container restarted${NC}"
    ;;

  logs)
    $DOCKER_COMMENTE logs -f water-meter-api
    ;;

  status)
    $DOCKER_COMPOSE ps
    ;;

  test)
    echo -e "${YELLOW}Testing API...${NC}"
    curl -X GET http://localhost:8000/health || echo -e "${RED}API not responding${NC}"
    ;;

  clean)
    echo -e "${YELLOW}Cleaning up Docker resources...${NC}"
    $DOCKER_COMPOSE down -v
    docker system prune -f
    echo -e "${GREEN}✓ Cleanup completed${NC}"
    ;;

  *)
    echo "Usage: $0 {build|run|stop|restart|logs|status|test|clean}"
    echo ""
    echo "Commands:"
    echo "  build   - Build Docker image"
    echo "  run     - Start container in detached mode"
    echo "  stop    - Stop and remove container"
    echo "  restart - Restart container"
    echo "  logs    - View container logs"
    echo "  status  - Show container status"
    echo "  test    - Test API health endpoint"
    echo "  clean   - Remove containers and unused Docker resources"
    echo ""
    echo "Quick start:"
    echo "  $0 build && $0 run"
    exit 1
    ;;
esac
