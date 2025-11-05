# TytoAlba Backend

Simple Go backend server that serves ship data from a text file.

## Features

- RESTful API endpoint for ship data
- Reads ship information from `data/ships.txt`
- CORS enabled for frontend integration
- Automatic ETA calculation

## Project Structure

```
backend/
├── cmd/
│   └── api/
│       └── main.go              # Main server entry point
├── internal/
│   └── handlers/
│       └── ships.go             # Ship data handlers
├── data/
│   └── ships.txt                # Ship data file
├── go.mod                       # Go module definition
└── README.md
```

## Installation

### Prerequisites

- Go 1.21 or higher

### Setup

```bash
cd backend

# Initialize Go module (if needed)
go mod tidy
```

## Running the Server

```bash
# From backend directory
go run cmd/api/main.go
```

The server will start on `http://localhost:8080`

## API Endpoints

### Get All Ships

```http
GET /api/ships
```

**Response Example:**
```json
[
  {
    "id": "1",
    "name": "Rasuna Baruna",
    "lat": -5.5,
    "lon": 112.5,
    "destination": "Taboneo Port",
    "destinationLat": -2.8,
    "destinationLon": 116.2,
    "eta": "2024-10-22T14:30:00Z",
    "estimatedFuel": 8500,
    "status": "In Progress",
    "route": [
      [-6.906, 110.831],
      [-5.5, 112.5],
      [-4.2, 114.8],
      [-2.8, 116.2]
    ],
    "currentRouteIndex": 1
  }
]
```

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "tytoalba-backend"
}
```

## Data Format

The `data/ships.txt` file uses pipe-delimited format:

```
id|name|current_lat|current_lon|destination|dest_lat|dest_lon|eta_hours|estimated_fuel|status|route_waypoints|current_route_index
```

**Example:**
```
1|Rasuna Baruna|-5.5|112.5|Taboneo Port|-2.8|116.2|24|8500|In Progress|-6.906,110.831;-5.5,112.5;-4.2,114.8;-2.8,116.2|1
```

**Fields:**
- `id`: Unique ship identifier
- `name`: Ship name
- `current_lat`, `current_lon`: Current position
- `destination`: Destination port name
- `dest_lat`, `dest_lon`: Destination coordinates
- `eta_hours`: Hours until arrival
- `estimated_fuel`: Estimated fuel on arrival (liters)
- `status`: Current status (In Progress, In Port, Delayed)
- `route_waypoints`: Semicolon-separated lat,lon pairs
- `current_route_index`: Index in route array where ship currently is

## Development

### Adding New Ships

Edit `data/ships.txt` and add new lines following the format above.

### Running with Auto-Reload

Install `air` for auto-reload during development:

```bash
go install github.com/cosmtrek/air@latest

# Run with air
air
```

## Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM golang:1.21-alpine AS builder

WORKDIR /app
COPY . .
RUN go build -o server cmd/api/main.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/server .
COPY data/ ./data/

EXPOSE 8080
CMD ["./server"]
```

Build and run:
```bash
docker build -t tytoalba-backend .
docker run -p 8080:8080 tytoalba-backend
```

## Testing

Test the API:

```bash
# Health check
curl http://localhost:8080/health

# Get ships
curl http://localhost:8080/api/ships
```

## CORS Configuration

CORS is enabled for all origins by default. For production, update the `Access-Control-Allow-Origin` header in `internal/handlers/ships.go` to specific domains:

```go
w.Header().Set("Access-Control-Allow-Origin", "https://your-frontend-domain.com")
```

## Troubleshooting

### Port Already in Use

Change the port in `cmd/api/main.go`:

```go
port := ":8081"  // Change from :8080
```

### Data File Not Found

Ensure you're running the server from the correct directory, or update the file path in `internal/handlers/ships.go`:

```go
ships, err := readShipsFromFile("path/to/your/data/ships.txt")
```

## Next Steps

- [ ] Add database integration (PostgreSQL/MongoDB)
- [ ] Implement authentication
- [ ] Add WebSocket support for real-time updates
- [ ] Integrate with ML service for predictions
- [ ] Add logging middleware
- [ ] Implement caching (Redis)

## License

MIT
