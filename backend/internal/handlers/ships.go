package handlers

import (
	"bufio"
	"encoding/json"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

type Waypoint struct {
	Lat float64 `json:"lat"`
	Lon float64 `json:"lon"`
}

type ShipData struct {
	ID                string      `json:"id"`
	Name              string      `json:"name"`
	Lat               float64     `json:"lat"`
	Lon               float64     `json:"lon"`
	Destination       string      `json:"destination"`
	DestinationLat    float64     `json:"destinationLat"`
	DestinationLon    float64     `json:"destinationLon"`
	ETA               string      `json:"eta"`
	EstimatedFuel     int         `json:"estimatedFuel"`
	Status            string      `json:"status"`
	Route             [][]float64 `json:"route"`
	CurrentRouteIndex int         `json:"currentRouteIndex"`
}

// GetShips reads ship data from text file and returns as JSON
func GetShips(w http.ResponseWriter, r *http.Request) {
	// Enable CORS
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.Header().Set("Content-Type", "application/json")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	ships, err := readShipsFromFile("backend/data/ships.txt")
	if err != nil {
		// Try alternative path
		ships, err = readShipsFromFile("data/ships.txt")
		if err != nil {
			http.Error(w, "Error reading ship data: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	json.NewEncoder(w).Encode(ships)
}

func readShipsFromFile(filepath string) ([]ShipData, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var ships []ShipData
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Skip comments and empty lines
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		ship, err := parseShipLine(line)
		if err != nil {
			continue // Skip invalid lines
		}

		ships = append(ships, ship)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return ships, nil
}

func parseShipLine(line string) (ShipData, error) {
	parts := strings.Split(line, "|")
	if len(parts) < 12 {
		return ShipData{}, nil
	}

	// Parse basic fields
	id := parts[0]
	name := parts[1]
	currentLat, _ := strconv.ParseFloat(parts[2], 64)
	currentLon, _ := strconv.ParseFloat(parts[3], 64)
	destination := parts[4]
	destLat, _ := strconv.ParseFloat(parts[5], 64)
	destLon, _ := strconv.ParseFloat(parts[6], 64)
	etaHours, _ := strconv.Atoi(parts[7])
	estimatedFuel, _ := strconv.Atoi(parts[8])
	status := parts[9]
	routeStr := parts[10]
	currentIndex, _ := strconv.Atoi(parts[11])

	// Calculate ETA
	eta := time.Now().Add(time.Duration(etaHours) * time.Hour)

	// Parse route waypoints
	route := parseRoute(routeStr)

	return ShipData{
		ID:                id,
		Name:              name,
		Lat:               currentLat,
		Lon:               currentLon,
		Destination:       destination,
		DestinationLat:    destLat,
		DestinationLon:    destLon,
		ETA:               eta.Format(time.RFC3339),
		EstimatedFuel:     estimatedFuel,
		Status:            status,
		Route:             route,
		CurrentRouteIndex: currentIndex,
	}, nil
}

func parseRoute(routeStr string) [][]float64 {
	var route [][]float64

	waypoints := strings.Split(routeStr, ";")
	for _, wp := range waypoints {
		coords := strings.Split(wp, ",")
		if len(coords) == 2 {
			lat, err1 := strconv.ParseFloat(strings.TrimSpace(coords[0]), 64)
			lon, err2 := strconv.ParseFloat(strings.TrimSpace(coords[1]), 64)
			if err1 == nil && err2 == nil {
				route = append(route, []float64{lat, lon})
			}
		}
	}

	return route
}
