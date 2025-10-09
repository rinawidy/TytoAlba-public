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
