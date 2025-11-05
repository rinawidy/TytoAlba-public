# Middleware Usage Example

## Before (Duplicated CORS and validation in every handler):

```go
func GetShipsFromMQTT(w http.ResponseWriter, r *http.Request) {
    // CORS headers (duplicated in every handler!)
    w.Header().Set("Access-Control-Allow-Origin", "*")
    w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
    w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
    w.Header().Set("Content-Type", "application/json")

    // OPTIONS handling (duplicated!)
    if r.Method == "OPTIONS" {
        w.WriteHeader(http.StatusOK)
        return
    }

    // ShipStore validation (duplicated!)
    if ShipStore == nil {
        http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
        return
    }

    // Actual handler logic
    ships := ShipStore.GetAllShips()
    json.NewEncoder(w).Encode(ships)
}
```

## After (Clean handler with middleware):

### Step 1: Simplify the handler

```go
func GetShipsFromMQTT(w http.ResponseWriter, r *http.Request) {
    // No CORS code needed!
    // No OPTIONS handling needed!
    // No ShipStore nil check needed!

    // Only business logic
    ships := ShipStore.GetAllShips()

    if err := json.NewEncoder(w).Encode(ships); err != nil {
        http.Error(w, `{"error":"Failed to encode ships data"}`, http.StatusInternalServerError)
        return
    }
}
```

### Step 2: Update main.go to use middleware

```go
package main

import (
    "net/http"

    "github.com/your-org/tytoalba/backend/internal/handlers"
    "github.com/your-org/tytoalba/backend/internal/middleware"
    "github.com/your-org/tytoalba/backend/internal/storage"
)

func main() {
    // Initialize ship store
    shipStore := storage.NewShipStore()
    handlers.ShipStore = shipStore

    // Create middleware chain
    withShipStore := middleware.WithShipStore(shipStore)

    // Register handlers with middleware
    http.HandleFunc("/api/mqtt/ships",
        middleware.Chain(
            handlers.GetShipsFromMQTT,
            middleware.WithCORS,
            withShipStore,
        ),
    )

    http.HandleFunc("/api/mqtt/bulk-carriers",
        middleware.Chain(
            handlers.GetBulkCarriers,
            middleware.WithCORS,
            withShipStore,
        ),
    )

    http.HandleFunc("/api/mqtt/ship",
        middleware.Chain(
            handlers.GetShipByMMSI,
            middleware.WithCORS,
            withShipStore,
        ),
    )

    http.HandleFunc("/api/mqtt/history",
        middleware.Chain(
            handlers.GetShipHistory,
            middleware.WithCORS,
            withShipStore,
        ),
    )

    http.HandleFunc("/api/mqtt/stats",
        middleware.Chain(
            handlers.GetShipStats,
            middleware.WithCORS,
            withShipStore,
        ),
    )

    // Start server
    http.ListenAndServe(":8080", nil)
}
```

## Benefits

### 1. Reduced Cyclomatic Complexity
- **Before**: Each handler has CC +2 (OPTIONS check + nil check)
- **After**: Middleware handles these, handlers focus on business logic
- **Reduction**: ~2 CC points per handler × 5 handlers = 10 CC points total

### 2. DRY Principle (Don't Repeat Yourself)
- CORS headers written once, not 5 times
- ShipStore validation once, not 5 times
- Easy to modify globally (change one place, affects all handlers)

### 3. Easier to Add New Features
Want to add authentication? Just add a middleware:

```go
http.HandleFunc("/api/mqtt/ships",
    middleware.Chain(
        handlers.GetShipsFromMQTT,
        middleware.WithCORS,
        middleware.WithAuth,  // NEW! Added in one place
        withShipStore,
    ),
)
```

### 4. Better Testing
- Test middleware independently
- Test handlers without worrying about CORS/validation
- Mock middleware for unit tests

### 5. Centralized Error Handling
All handlers get consistent error responses without duplicating error handling logic.

## Cyclomatic Complexity Comparison

### GetShipsFromMQTT

**Before:**
```
CC = 1 (base) + 1 (OPTIONS) + 1 (ShipStore nil) + 1 (encode error) = 4
```

**After:**
```
CC = 1 (base) + 1 (encode error) = 2
```

**Saved: 2 CC points per handler**

### All 5 Handlers Combined

**Before Total CC:**
- GetShipsFromMQTT: 4
- GetBulkCarriers: 4
- GetShipByMMSI: 6
- GetShipHistory: 5 (already refactored from 7)
- GetShipStats: 4
**Total: 23**

**After Total CC:**
- GetShipsFromMQTT: 2
- GetBulkCarriers: 2
- GetShipByMMSI: 4
- GetShipHistory: 3
- GetShipStats: 2
**Total: 13**

**Total Reduction: 10 CC points (43% improvement!)**

## Next Steps

1. Update all 5 handlers to remove CORS and validation code
2. Update cmd/api/main.go to use middleware.Chain()
3. Run tests to ensure functionality unchanged
4. Measure CC with gocyclo to confirm reduction
