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
