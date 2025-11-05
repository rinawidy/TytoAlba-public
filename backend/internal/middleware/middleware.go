package middleware

import (
	"net/http"

	"github.com/your-org/tytoalba/backend/internal/storage"
)

// WithCORS adds CORS headers to all responses
func WithCORS(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		w.Header().Set("Content-Type", "application/json")

		// Handle preflight requests
		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Call the next handler
		next(w, r)
	}
}

// WithShipStore validates that ShipStore is initialized before processing
func WithShipStore(shipStore *storage.ShipStore) func(http.HandlerFunc) http.HandlerFunc {
	return func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if shipStore == nil {
				http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
				return
			}

			// Call the next handler
			next(w, r)
		}
	}
}

// Chain combines multiple middleware functions
func Chain(h http.HandlerFunc, middlewares ...func(http.HandlerFunc) http.HandlerFunc) http.HandlerFunc {
	for i := len(middlewares) - 1; i >= 0; i-- {
		h = middlewares[i](h)
	}
	return h
}
