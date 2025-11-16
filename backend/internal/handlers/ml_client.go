package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"time"
)

// MLServiceConfig holds ML service configuration
type MLServiceConfig struct {
	BaseURL string
	Timeout time.Duration
}

// ArrivalPredictionRequest matches ML service schema
type ArrivalPredictionRequest struct {
	Distance           float64  `json:"distance"`
	DepartureTime      string   `json:"departure_time,omitempty"`
	DepartureHour      *int     `json:"departure_hour,omitempty"`
	DayOfWeek          *int     `json:"day_of_week,omitempty"`
	RouteID            int      `json:"route_id"`
	AvgTrafficLevel    int      `json:"avg_traffic_level"`
	HistoricalAvgTime  *float64 `json:"historical_avg_time,omitempty"`
}

// ArrivalPredictionResponse matches ML service response
type ArrivalPredictionResponse struct {
	TravelTimeMinutes    float64 `json:"travel_time_minutes"`
	EstimatedArrivalTime string  `json:"estimated_arrival_time,omitempty"`
	FeaturesUsed         map[string]interface{} `json:"features_used"`
}

// MLHealthResponse matches ML service health check
type MLHealthResponse struct {
	Status              string `json:"status"`
	FuelModelLoaded     bool   `json:"fuel_model_loaded"`
	ArrivalModelLoaded  bool   `json:"arrival_model_loaded"`
}

// getMLServiceConfig returns ML service configuration
func getMLServiceConfig() MLServiceConfig {
	baseURL := os.Getenv("ML_SERVICE_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}

	return MLServiceConfig{
		BaseURL: baseURL,
		Timeout: 10 * time.Second,
	}
}

// checkMLServiceHealth checks if ML service is healthy
func checkMLServiceHealth() (bool, error) {
	config := getMLServiceConfig()
	client := &http.Client{Timeout: config.Timeout}

	resp, err := client.Get(config.BaseURL + "/health")
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("ML service returned status %d", resp.StatusCode)
	}

	var healthResp MLHealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&healthResp); err != nil {
		return false, err
	}

	return healthResp.Status == "healthy" && healthResp.ArrivalModelLoaded, nil
}

// predictArrivalTime calls ML service to predict arrival time
func predictArrivalTime(distance float64, routeID int, historicalAvgTime *float64) (*ArrivalPredictionResponse, error) {
	config := getMLServiceConfig()
	client := &http.Client{Timeout: config.Timeout}

	// Prepare request
	now := time.Now()
	request := ArrivalPredictionRequest{
		Distance:          distance,
		DepartureTime:     now.Format(time.RFC3339),
		RouteID:           routeID,
		AvgTrafficLevel:   1, // Default to medium traffic
		HistoricalAvgTime: historicalAvgTime,
	}

	// Marshal request to JSON
	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Make POST request to ML service
	resp, err := client.Post(
		config.BaseURL+"/predict/arrival",
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to call ML service: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ML service returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var predictionResp ArrivalPredictionResponse
	if err := json.Unmarshal(body, &predictionResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &predictionResp, nil
}

// haversineDistance calculates distance between two lat/lon points in kilometers
func haversineDistance(lat1, lon1, lat2, lon2 float64) float64 {
	const earthRadiusKm = 6371.0

	// Convert to radians
	lat1Rad := lat1 * math.Pi / 180
	lat2Rad := lat2 * math.Pi / 180
	deltaLat := (lat2 - lat1) * math.Pi / 180
	deltaLon := (lon2 - lon1) * math.Pi / 180

	// Haversine formula
	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadiusKm * c
}

// encodeRouteID creates a route ID from port code
func encodeRouteID(portCode string) int {
	// Simple hash function for port codes
	hash := 0
	for _, char := range portCode {
		hash = (hash * 31) + int(char)
	}
	// Ensure positive and within reasonable range
	return int(math.Abs(float64(hash))) % 10000
}

// calculateHistoricalAvgTime calculates average travel time from historical data
func calculateHistoricalAvgTime(mmsi string, destination string) *float64 {
	// Load historical data
	historicalData, err := loadHistoricalData("backend/data/historical_voyages_15min.json")
	if err != nil {
		historicalData, err = loadHistoricalData("data/historical_voyages_15min.json")
		if err != nil {
			return nil
		}
	}

	// Filter records for this ship and destination
	var relevantRecords []HistoricalRecord
	for _, record := range historicalData.Data {
		if record.MMSI == mmsi && record.Destination == destination {
			relevantRecords = append(relevantRecords, record)
		}
	}

	// If we have at least 2 records, calculate average time between them
	if len(relevantRecords) >= 2 {
		// Parse timestamps and calculate time differences
		totalMinutes := 0.0
		count := 0

		for i := 1; i < len(relevantRecords); i++ {
			t1, err1 := time.Parse("2006-01-02 15:04:05", relevantRecords[i-1].Timestamp)
			t2, err2 := time.Parse("2006-01-02 15:04:05", relevantRecords[i].Timestamp)

			if err1 == nil && err2 == nil {
				diff := t2.Sub(t1).Minutes()
				if diff > 0 {
					totalMinutes += diff
					count++
				}
			}
		}

		if count > 0 {
			avgTime := totalMinutes / float64(count)
			return &avgTime
		}
	}

	return nil
}
