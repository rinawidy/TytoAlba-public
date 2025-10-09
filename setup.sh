#!/bin/bash

# TytoAlba - Complete Project Setup Script
# Run this to recreate the entire project structure

set -e  # Exit on error

echo "🚀 Setting up TytoAlba - Fuel Management System"
echo "================================================"
echo ""

# Create complete directory structure
echo "📁 Creating directory structure..."

mkdir -p frontend/{src/{components/{dashboard,map,charts,tables,predictions},views,services,stores,types,assets/styles,utils,composables},public}
mkdir -p backend/{cmd/api,internal/{handlers,services,models,repository/{postgres,redis},middleware,config},pkg/{utils,logger},migrations}
mkdir -p ml-service/{models,src}
mkdir -p infrastructure/{nginx/conf.d,kubernetes/{deployments,services,configmaps,secrets,ingress}}
mkdir -p database/{migrations,seeds}
mkdir -p docker scripts docs .vscode

echo "✅ Directory structure created!"
echo ""

# Create .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
# Dependencies
node_modules/
vendor/

# Build outputs
dist/
build/
bin/
*.exe

# Environment variables
.env
.env.local

# IDE
.vscode/*
!.vscode/settings.json
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Database
*.db
*.sqlite

# Docker
docker-compose.override.yml

# Go
*.test
*.out
coverage.txt

# Vue
.nuxt/
.output/
EOF

echo "✅ .gitignore created!"
echo ""

# Create .env.example
echo "📝 Creating .env.example..."
cat > .env.example << 'EOF'
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=fms_user
DB_PASSWORD=fms_password_2025
DB_NAME=fms_db

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis_password_2025

# Backend Configuration
PORT=8000
JWT_SECRET=your-jwt-secret-change-in-production
GIN_MODE=debug

# Frontend Configuration
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000

# ML Service
ML_SERVICE_URL=http://localhost:8001
EOF

cp .env.example .env
echo "✅ Environment files created!"
echo ""

# Create VSCode settings
echo "📝 Creating VSCode settings..."
cat > .vscode/settings.json << 'EOF'
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[go]": {
    "editor.defaultFormatter": "golang.go"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[vue]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "go.useLanguageServer": true,
  "go.lintTool": "golangci-lint",
  "files.associations": {
    "*.vue": "vue"
  },
  "typescript.tsdk": "node_modules/typescript/lib",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  }
}
EOF

echo "✅ VSCode settings created!"
echo ""

# Create README.md
echo "📝 Creating README.md..."
cat > README.md << 'EOF'
# TytoAlba - Fuel Management System

Ship tracking and fuel consumption prediction system.

## Tech Stack

- **Frontend**: Vue 3 + TypeScript + Vite + Tailwind CSS
- **Backend**: Golang + Gin + GORM
- **Database**: PostgreSQL 15
- **Cache**: Redis 7
- **Reverse Proxy**: Nginx
- **Containerization**: Docker + Kubernetes

## Quick Start

```bash
# Install dependencies
make install

# Start all services with Docker
make dev-docker

# Or start development servers
make dev
```

## Access Points

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/health
- Nginx: http://localhost:80

## Available Commands

```bash
make help           # Show all commands
make install        # Install dependencies
make dev            # Start dev servers
make dev-docker     # Start with Docker
make up             # Start services
make down           # Stop services
make logs           # View logs
make migrate        # Run migrations
make test           # Run tests
make clean          # Clean up
```

## Project Structure

```
TytoAlba/
├── frontend/          # Vue 3 + TypeScript
├── backend/           # Golang API
├── database/          # PostgreSQL schemas
├── infrastructure/    # Nginx & K8s configs
├── ml-service/        # Python ML service
└── docker/            # Docker configs
```

## Documentation

- [Setup Guide](docs/SETUP.md)
- [API Documentation](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)

## Features

- 🚢 Real-time ship tracking
- ⛽ Fuel consumption monitoring
- 📊 Data visualization
- 🔮 Fuel consumption prediction
- ⏱️ Arrival time estimation
- 📱 Responsive dashboard

## License

MIT
EOF

echo "✅ README.md created!"
echo ""

# Create docker-compose.yml
echo "📝 Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15-alpine
    container_name: tytoalba_postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: fms_db
      POSTGRES_USER: fms_user
      POSTGRES_PASSWORD: fms_password_2025
      TZ: Asia/Jakarta
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - tytoalba_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U fms_user -d fms_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: tytoalba_redis
    restart: unless-stopped
    command: redis-server --appendonly yes --requirepass redis_password_2025
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - tytoalba_network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: tytoalba_backend
    restart: unless-stopped
    environment:
      DB_HOST: postgres
      DB_PORT: 5432
      DB_USER: fms_user
      DB_PASSWORD: fms_password_2025
      DB_NAME: fms_db
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_PASSWORD: redis_password_2025
      JWT_SECRET: your-jwt-secret-change-in-production
      PORT: 8000
      GIN_MODE: release
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - tytoalba_network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        VITE_API_URL: http://localhost:8000
        VITE_WS_URL: ws://localhost:8000
    container_name: tytoalba_frontend
    restart: unless-stopped
    environment:
      NODE_ENV: production
      VITE_API_URL: http://localhost:8000
      VITE_WS_URL: ws://localhost:8000
    ports:
      - "3000:3000"
    depends_on:
      - backend
    networks:
      - tytoalba_network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: tytoalba_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./infrastructure/nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./infrastructure/nginx/conf.d:/etc/nginx/conf.d:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - frontend
      - backend
    networks:
      - tytoalba_network
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

networks:
  tytoalba_network:
    driver: bridge

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  nginx_logs:
    driver: local
EOF

echo "✅ docker-compose.yml created!"
echo ""

# Create Makefile
echo "📝 Creating Makefile..."
cat > Makefile << 'EOF'
.PHONY: help setup install dev up down logs clean

GREEN  := $(shell tput -Txterm setaf 2)
YELLOW := $(shell tput -Txterm setaf 3)
RESET  := $(shell tput -Txterm sgr0)

help:
	@echo '${GREEN}TytoAlba - Fuel Management System${RESET}'
	@echo ''
	@echo 'Usage:'
	@echo '  ${YELLOW}make${RESET} ${GREEN}<target>${RESET}'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-15s${RESET} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependencies
	@echo "${GREEN}Installing dependencies...${RESET}"
	cd frontend && npm install
	cd backend && go mod download
	@echo "${GREEN}✅ Dependencies installed!${RESET}"

dev: ## Run development servers
	@echo "${GREEN}Starting development servers...${RESET}"
	docker-compose up -d postgres redis
	@echo "${YELLOW}Starting backend...${RESET}"
	cd backend && go run cmd/api/main.go &
	@echo "${YELLOW}Starting frontend...${RESET}"
	cd frontend && npm run dev

dev-docker: ## Run all services with Docker
	@echo "${GREEN}Starting all services with Docker...${RESET}"
	docker-compose up --build -d
	@echo "${GREEN}✅ All services started!${RESET}"

up: ## Start all services
	docker-compose up -d

down: ## Stop all services
	docker-compose down

restart: down up ## Restart all services

logs: ## View all logs
	docker-compose logs -f

logs-backend: ## View backend logs
	docker-compose logs -f backend

logs-frontend: ## View frontend logs
	docker-compose logs -f frontend

migrate: ## Run database migrations
	@echo "${GREEN}Running migrations...${RESET}"
	docker-compose exec postgres psql -U fms_user -d fms_db -f /docker-entrypoint-initdb.d/init.sql

test: ## Run all tests
	cd backend && go test ./...
	cd frontend && npm run test

clean: ## Clean up containers and volumes
	docker-compose down -v
	rm -rf frontend/node_modules frontend/dist
	rm -rf backend/bin

build: ## Build all services
	docker-compose build
EOF

echo "✅ Makefile created!"
echo ""

echo "================================================"
echo "✅ TytoAlba project structure created successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Initialize frontend: cd frontend && npm create vite@latest . -- --template vue-ts"
echo "2. Initialize backend: cd backend && go mod init tytoalba/backend"
echo "3. Install dependencies: make install"
echo "4. Start services: make dev-docker"
echo ""
echo "🚀 Happy coding!"
echo "================================================"