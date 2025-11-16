package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"time"
)

// HistoricalRecord represents a single historical position record
type HistoricalRecord struct {
	MMSI        string  `json:"mmsi"`
	VesselName  string  `json:"shipname"` // JSON uses "shipname" not "vessel_name"
	Latitude    float64 `json:"latitude"`
	Longitude   float64 `json:"longitude"`
	SpeedKnots  float64 `json:"speed_knots"`
	Course      int     `json:"course"`
	Timestamp   string  `json:"timestamp"`
	LastPort    string  `json:"last_port,omitempty"`
	Destination string  `json:"destination,omitempty"`
	ETA         string  `json:"eta,omitempty"`
}

// HistoricalData represents the full historical dataset
type HistoricalData struct {
	Metadata struct {
		GeneratedAt    string `json:"generated_at"`
		IntervalMinutes int   `json:"interval_minutes"`
		TotalRecords   int    `json:"total_records"`
		TotalShips     int    `json:"total_ships"`
		TimeSpanStart  string `json:"time_span_start"`
		TimeSpanEnd    string `json:"time_span_end"`
		LastUpdated    string `json:"last_updated,omitempty"`
	} `json:"metadata"`
	Data []HistoricalRecord `json:"data"`
}

// GetHistoricalVoyages returns all historical voyage data
// GET /api/historical/voyages
func GetHistoricalVoyages(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Load historical data
	historicalData, err := loadHistoricalData("backend/data/historical_voyages_15min.json")
	if err != nil {
		// Try alternative path
		historicalData, err = loadHistoricalData("data/historical_voyages_15min.json")
		if err != nil {
			http.Error(w, "Error reading historical data: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	json.NewEncoder(w).Encode(historicalData)
}

// GetVesselHistory returns historical data for a specific vessel
// GET /api/historical/vessel?mmsi=123456789
func GetVesselHistory(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Get MMSI from query parameter
	mmsi := r.URL.Query().Get("mmsi")
	if mmsi == "" {
		http.Error(w, "MMSI parameter is required", http.StatusBadRequest)
		return
	}

	// Load historical data
	historicalData, err := loadHistoricalData("backend/data/historical_voyages_15min.json")
	if err != nil {
		historicalData, err = loadHistoricalData("data/historical_voyages_15min.json")
		if err != nil {
			http.Error(w, "Error reading historical data: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	// Filter records for specific MMSI
	var vesselRecords []HistoricalRecord
	for _, record := range historicalData.Data {
		if record.MMSI == mmsi {
			vesselRecords = append(vesselRecords, record)
		}
	}

	if len(vesselRecords) == 0 {
		http.Error(w, "No historical data found for MMSI: "+mmsi, http.StatusNotFound)
		return
	}

	response := map[string]interface{}{
		"mmsi":          mmsi,
		"vessel_name":   vesselRecords[0].VesselName,
		"record_count":  len(vesselRecords),
		"time_span_start": vesselRecords[0].Timestamp,
		"time_span_end":   vesselRecords[len(vesselRecords)-1].Timestamp,
		"records":       vesselRecords,
	}

	json.NewEncoder(w).Encode(response)
}

// GetHistoricalStats returns statistics about historical data
// GET /api/historical/stats
func GetHistoricalStats(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Load historical data
	historicalData, err := loadHistoricalData("backend/data/historical_voyages_15min.json")
	if err != nil {
		historicalData, err = loadHistoricalData("data/historical_voyages_15min.json")
		if err != nil {
			http.Error(w, "Error reading historical data: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	// Calculate statistics
	stats := calculateHistoricalStats(historicalData)

	json.NewEncoder(w).Encode(stats)
}

// GetRecentPositions returns recent positions (last N hours)
// GET /api/historical/recent?hours=24
func GetRecentPositions(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Get hours from query parameter (default: 24)
	hoursStr := r.URL.Query().Get("hours")
	hours := 24
	if hoursStr != "" {
		if h, err := strconv.Atoi(hoursStr); err == nil {
			hours = h
		}
	}

	// Load historical data
	historicalData, err := loadHistoricalData("backend/data/historical_voyages_15min.json")
	if err != nil {
		historicalData, err = loadHistoricalData("data/historical_voyages_15min.json")
		if err != nil {
			http.Error(w, "Error reading historical data: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	// Calculate cutoff time
	cutoffTime := time.Now().Add(-time.Duration(hours) * time.Hour)

	// Filter recent records
	var recentRecords []HistoricalRecord
	for _, record := range historicalData.Data {
		recordTime, err := time.Parse("2006-01-02 15:04:05", record.Timestamp)
		if err != nil {
			continue
		}

		if recordTime.After(cutoffTime) {
			recentRecords = append(recentRecords, record)
		}
	}

	response := map[string]interface{}{
		"hours":        hours,
		"cutoff_time":  cutoffTime.Format("2006-01-02 15:04:05"),
		"record_count": len(recentRecords),
		"records":      recentRecords,
	}

	json.NewEncoder(w).Encode(response)
}

// loadHistoricalData loads historical voyage data from JSON file
func loadHistoricalData(filepath string) (*HistoricalData, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data HistoricalData
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, err
	}

	return &data, nil
}

// calculateHistoricalStats calculates statistics from historical data
func calculateHistoricalStats(data *HistoricalData) map[string]interface{} {
	// Count vessels
	vesselCount := make(map[string]bool)
	recordsByVessel := make(map[string]int)

	for _, record := range data.Data {
		vesselCount[record.MMSI] = true
		recordsByVessel[record.MMSI]++
	}

	// Find vessel with most records
	maxRecords := 0
	maxRecordsVessel := ""
	for mmsi, count := range recordsByVessel {
		if count > maxRecords {
			maxRecords = count
			maxRecordsVessel = mmsi
		}
	}

	// Find vessel name for max records
	maxRecordsVesselName := ""
	for _, record := range data.Data {
		if record.MMSI == maxRecordsVessel {
			maxRecordsVesselName = record.VesselName
			break
		}
	}

	return map[string]interface{}{
		"total_records":        data.Metadata.TotalRecords,
		"total_vessels":        len(vesselCount),
		"time_span_start":      data.Metadata.TimeSpanStart,
		"time_span_end":        data.Metadata.TimeSpanEnd,
		"interval_minutes":     data.Metadata.IntervalMinutes,
		"last_updated":         data.Metadata.LastUpdated,
		"vessel_with_most_data": map[string]interface{}{
			"mmsi":         maxRecordsVessel,
			"vessel_name":  maxRecordsVesselName,
			"record_count": maxRecords,
		},
		"records_per_vessel":   recordsByVessel,
	}
}
