package handlers

import (
	"encoding/json"
	"net/http"
	"strconv"

	"github.com/your-org/tytoalba/backend/internal/storage"
)

var ShipStore *storage.ShipStore

// GetShipsFromMQTT returns all ships from MQTT data store
func GetShipsFromMQTT(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if ShipStore == nil {
		http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
		return
	}

	ships := ShipStore.GetAllShips()

	if err := json.NewEncoder(w).Encode(ships); err != nil {
		http.Error(w, `{"error":"Failed to encode ships data"}`, http.StatusInternalServerError)
		return
	}
}

// GetBulkCarriers returns only bulk carrier ships
func GetBulkCarriers(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if ShipStore == nil {
		http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
		return
	}

	ships := ShipStore.GetBulkCarriers()

	if err := json.NewEncoder(w).Encode(ships); err != nil {
		http.Error(w, `{"error":"Failed to encode ships data"}`, http.StatusInternalServerError)
		return
	}
}

// GetShipByMMSI returns data for a specific ship
func GetShipByMMSI(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if ShipStore == nil {
		http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
		return
	}

	// Get MMSI from query parameter
	mmsi := r.URL.Query().Get("mmsi")
	if mmsi == "" {
		http.Error(w, `{"error":"MMSI parameter required"}`, http.StatusBadRequest)
		return
	}

	ship, exists := ShipStore.GetShip(mmsi)
	if !exists {
		http.Error(w, `{"error":"Ship not found"}`, http.StatusNotFound)
		return
	}

	if err := json.NewEncoder(w).Encode(ship); err != nil {
		http.Error(w, `{"error":"Failed to encode ship data"}`, http.StatusInternalServerError)
		return
	}
}

// GetShipHistory returns historical data for a ship
func GetShipHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if ShipStore == nil {
		http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
		return
	}

	// Get parameters
	mmsi := r.URL.Query().Get("mmsi")
	if mmsi == "" {
		http.Error(w, `{"error":"MMSI parameter required"}`, http.StatusBadRequest)
		return
	}

	hours := parseHoursParameter(r.URL.Query().Get("hours"), 24)

	history := ShipStore.GetShipHistory(mmsi, hours)

	if err := json.NewEncoder(w).Encode(history); err != nil {
		http.Error(w, `{"error":"Failed to encode history data"}`, http.StatusInternalServerError)
		return
	}
}

// parseHoursParameter parses the hours query parameter with a default fallback
func parseHoursParameter(hoursStr string, defaultValue int) int {
	if hoursStr == "" {
		return defaultValue
	}

	if h, err := strconv.Atoi(hoursStr); err == nil && h > 0 {
		return h
	}

	return defaultValue
}

// GetShipStats returns statistics about tracked ships
func GetShipStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	if ShipStore == nil {
		http.Error(w, `{"error":"Ship store not initialized"}`, http.StatusInternalServerError)
		return
	}

	stats := map[string]interface{}{
		"total_ships": ShipStore.GetShipCount(),
		"by_type":     ShipStore.GetShipCountByType(),
	}

	if err := json.NewEncoder(w).Encode(stats); err != nil {
		http.Error(w, `{"error":"Failed to encode stats"}`, http.StatusInternalServerError)
		return
	}
}
