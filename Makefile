.PHONY: help build run stop restart logs status test clean shell

help: ## Show this help message
	@echo 'Water Meter AI - Docker Management'
	@echo ''
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-12s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build: ## Build Docker image
	docker-compose build --no-cache
	@echo "✓ Build completed"

run: ## Start container in detached mode
	docker-compose up -d
	@echo "✓ Container started"
	@echo "API available at: http://localhost:8000"
	@echo "API docs at: http://localhost:8000/docs"

stop: ## Stop and remove container
	docker-compose down
	@echo "✓ Container stopped"

restart: ## Restart container
	docker-compose restart
	@echo "✓ Container restarted"

logs: ## View container logs (live)
	docker-compose logs -f water-meter-api

status: ## Show container status
	docker-compose ps

test: ## Test API health endpoint
	@curl -s http://localhost:8000/health || echo "✗ API not responding"

clean: stop ## Remove containers and unused Docker resources
	docker system prune -f
	@echo "✓ Cleanup completed"

shell: ## Open shell in running container
	docker-compose exec water-meter-api /bin/bash

rebuild: clean build ## Clean rebuild (clean + build)
	@echo "✓ Rebuild completed"

dev: ## Run in development mode with hot reload
	docker-compose up --build

inspect: ## Show container detailed info
	docker inspect water-meter-api

stats: ## Show container resource usage
	docker stats water-meter-api

.DEFAULT_GOAL := help
