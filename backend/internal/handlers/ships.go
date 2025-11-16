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
	MMSI              string      `json:"mmsi"`
	Name              string      `json:"name"`
	Type              string      `json:"type"`
	CoalCapacity      int         `json:"coalCapacity"`
	LOA               float64     `json:"loa"`
	Beam              float64     `json:"beam"`
	DWT               int         `json:"dwt"`
	Lat               float64     `json:"lat"`
	Lon               float64     `json:"lon"`
	Destination       string      `json:"destination,omitempty"`
	DestinationLat    float64     `json:"destinationLat,omitempty"`
	DestinationLon    float64     `json:"destinationLon,omitempty"`
	ETA               string      `json:"eta,omitempty"`
	EstimatedFuel     int         `json:"estimatedFuel,omitempty"`
	Status            string      `json:"status"`
	Route             [][]float64 `json:"route,omitempty"`
	CurrentRouteIndex int         `json:"currentRouteIndex,omitempty"`
	HistoricalTrail   [][]float64 `json:"historicalTrail,omitempty"` // Past positions
	PushingBarge      string      `json:"pushingBarge,omitempty"`
	BargeCoalCapacity int         `json:"bargeCoalCapacity,omitempty"`
	PushedBy          string      `json:"pushedBy,omitempty"`
}

type PositionData struct {
	Latitude    float64 `json:"latitude"`
	Longitude   float64 `json:"longitude"`
	SpeedKnots  float64 `json:"speed_knots"`
	Course      int     `json:"course"`
	Timestamp   string  `json:"timestamp"`
	LastUpdated string  `json:"last_updated"`
}

type VoyageData struct {
	LastPort    string `json:"last_port"`
	Destination string `json:"destination"`
	ETA         string `json:"eta"`
	Status      string `json:"status"`
}

type ShipMaster struct {
	VesselMMSI      string        `json:"vessel_mmsi"`
	VesselName      string        `json:"vessel_name"`
	ShipType        string        `json:"ship_type"`
	VesselType      string        `json:"vessel_type"`
	LOA             float64       `json:"loa"`
	Beam            float64       `json:"beam"`
	DWT             int           `json:"deadweight_tonnage"`
	CoalCapacity    int           `json:"coal_capacity"`
	BargeMMSI       string        `json:"barge_mmsi,omitempty"`
	TugboatMMSI     string        `json:"tugboat_mmsi,omitempty"`
	CurrentPosition *PositionData `json:"current_position,omitempty"`
	CurrentVoyage   *VoyageData   `json:"current_voyage,omitempty"`
}

type ShipMasterData struct {
	BulkCarriers []ShipMaster `json:"bulk_carriers"`
	Tugboats     []ShipMaster `json:"tugboats"`
	Barges       []ShipMaster `json:"barges"`
}

// GetShips reads ship data from text file, merges with master data, and returns as JSON
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

	// Load ship master data
	masterData, err := loadShipMasterData("backend/data/ships_master.json")
	if err != nil {
		// Try alternative path
		masterData, err = loadShipMasterData("data/ships_master.json")
		if err != nil {
			http.Error(w, "Error reading ship master data: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}

	// Load position data from ships.txt
	positionData, err := readShipsFromFile("backend/data/ships.txt")
	if err != nil {
		// Try alternative path
		positionData, err = readShipsFromFile("data/ships.txt")
		// If ships.txt doesn't exist, just use master data without positions
		if err != nil {
			positionData = []ShipData{}
		}
	}

	// Merge master data with position data
	ships := mergeShipData(masterData, positionData)

	// Enrich with historical trails and calculated ETA
	enrichShipsWithHistoricalData(&ships)

	// Generate predicted routes for ships with destinations
	generatePredictedRoutes(&ships, masterData)

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

// loadShipMasterData loads ship specifications from ships_master.json
func loadShipMasterData(filepath string) (*ShipMasterData, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var masterData ShipMasterData
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&masterData); err != nil {
		return nil, err
	}

	return &masterData, nil
}

// mergeShipData merges ship specifications from master data with position data
func mergeShipData(masterData *ShipMasterData, positionData []ShipData) []ShipData {
	var ships []ShipData
	positionMap := make(map[string]ShipData)

	// Create map of position data by ship name
	for _, pos := range positionData {
		positionMap[pos.Name] = pos
	}

	// Counter for IDs
	id := 1

	// Process bulk carriers
	for _, master := range masterData.BulkCarriers {
		ship := ShipData{
			ID:           strconv.Itoa(id),
			MMSI:         master.VesselMMSI,
			Name:         master.VesselName,
			Type:         master.VesselType,
			CoalCapacity: master.CoalCapacity,
			LOA:          master.LOA,
			Beam:         master.Beam,
			DWT:          master.DWT,
			Status:       "active",
		}

		// Use position data from ships_master.json if available
		if master.CurrentPosition != nil {
			ship.Lat = master.CurrentPosition.Latitude
			ship.Lon = master.CurrentPosition.Longitude
		}

		// Use voyage data from ships_master.json if available
		if master.CurrentVoyage != nil {
			ship.Destination = master.CurrentVoyage.Destination
			ship.ETA = master.CurrentVoyage.ETA
			ship.Status = master.CurrentVoyage.Status
		}

		// Merge with position data from ships.txt if available (legacy support)
		if pos, exists := positionMap[master.VesselName]; exists {
			if pos.Lat != 0 {
				ship.Lat = pos.Lat
			}
			if pos.Lon != 0 {
				ship.Lon = pos.Lon
			}
			if pos.Destination != "" {
				ship.Destination = pos.Destination
			}
			ship.DestinationLat = pos.DestinationLat
			ship.DestinationLon = pos.DestinationLon
			if pos.ETA != "" {
				ship.ETA = pos.ETA
			}
			ship.EstimatedFuel = pos.EstimatedFuel
			if pos.Status != "" {
				ship.Status = pos.Status
			}
			ship.Route = pos.Route
			ship.CurrentRouteIndex = pos.CurrentRouteIndex
		}

		ships = append(ships, ship)
		id++
	}

	// Process tugboats
	for _, master := range masterData.Tugboats {
		ship := ShipData{
			ID:           strconv.Itoa(id),
			MMSI:         master.VesselMMSI,
			Name:         master.VesselName,
			Type:         master.VesselType,
			CoalCapacity: master.CoalCapacity,
			LOA:          master.LOA,
			Beam:         master.Beam,
			DWT:          master.DWT,
			Status:       "active",
		}

		// Use position data from ships_master.json if available
		if master.CurrentPosition != nil {
			ship.Lat = master.CurrentPosition.Latitude
			ship.Lon = master.CurrentPosition.Longitude
		}

		// Use voyage data from ships_master.json if available
		if master.CurrentVoyage != nil {
			ship.Destination = master.CurrentVoyage.Destination
			ship.ETA = master.CurrentVoyage.ETA
			ship.Status = master.CurrentVoyage.Status
		}

		// Add tugboat-specific fields
		if master.BargeMMSI != "" {
			ship.PushingBarge = findVesselNameByMMSI(masterData, master.BargeMMSI)
			bargeCapacity := findBargeCoalCapacity(masterData, master.BargeMMSI)
			ship.BargeCoalCapacity = bargeCapacity
		}

		ships = append(ships, ship)
		id++
	}

	// Process barges
	for _, master := range masterData.Barges {
		ship := ShipData{
			ID:           strconv.Itoa(id),
			MMSI:         master.VesselMMSI,
			Name:         master.VesselName,
			Type:         master.VesselType,
			CoalCapacity: master.CoalCapacity,
			LOA:          master.LOA,
			Beam:         master.Beam,
			DWT:          master.DWT,
			Status:       "active",
		}

		// Use position data from ships_master.json if available
		if master.CurrentPosition != nil {
			ship.Lat = master.CurrentPosition.Latitude
			ship.Lon = master.CurrentPosition.Longitude
		}

		// Use voyage data from ships_master.json if available
		if master.CurrentVoyage != nil {
			ship.Destination = master.CurrentVoyage.Destination
			ship.ETA = master.CurrentVoyage.ETA
			ship.Status = master.CurrentVoyage.Status
		}

		// Add barge-specific fields
		if master.TugboatMMSI != "" {
			ship.PushedBy = findVesselNameByMMSI(masterData, master.TugboatMMSI)
		}

		ships = append(ships, ship)
		id++
	}

	return ships
}

// Helper function to find vessel name by MMSI
func findVesselNameByMMSI(masterData *ShipMasterData, mmsi string) string {
	for _, v := range masterData.Barges {
		if v.VesselMMSI == mmsi {
			return v.VesselName
		}
	}
	for _, v := range masterData.Tugboats {
		if v.VesselMMSI == mmsi {
			return v.VesselName
		}
	}
	return ""
}

// Helper function to find barge coal capacity
func findBargeCoalCapacity(masterData *ShipMasterData, mmsi string) int {
	for _, v := range masterData.Barges {
		if v.VesselMMSI == mmsi {
			return v.CoalCapacity
		}
	}
	return 0
}

// normalizeShipName normalizes ship name for matching (removes "MV.", uppercase, trim)
func normalizeShipName(name string) string {
	normalized := strings.ToUpper(strings.TrimSpace(name))
	normalized = strings.ReplaceAll(normalized, "MV.", "")
	normalized = strings.ReplaceAll(normalized, "M.V.", "")
	normalized = strings.TrimSpace(normalized)
	return normalized
}

// enrichShipsWithHistoricalData adds historical trails and calculates real ETA
func enrichShipsWithHistoricalData(ships *[]ShipData) {
	// Load historical data
	historicalData, err := loadHistoricalData("backend/data/historical_voyages_15min.json")
	if err != nil {
		historicalData, err = loadHistoricalData("data/historical_voyages_15min.json")
		if err != nil {
			// Historical data not available, skip enrichment
			return
		}
	}

	// Create maps of historical records by MMSI and normalized ship name
	historicalMapByMMSI := make(map[string][]HistoricalRecord)
	historicalMapByName := make(map[string][]HistoricalRecord)

	for _, record := range historicalData.Data {
		// Index by MMSI
		historicalMapByMMSI[record.MMSI] = append(historicalMapByMMSI[record.MMSI], record)

		// Index by normalized ship name
		normalizedName := normalizeShipName(record.VesselName)
		if normalizedName != "" {
			historicalMapByName[normalizedName] = append(historicalMapByName[normalizedName], record)
		}
	}

	// Enrich each ship
	for i := range *ships {
		ship := &(*ships)[i]

		// Try to get historical records by MMSI first, then by ship name
		var records []HistoricalRecord
		var exists bool

		// Try MMSI match
		if ship.MMSI != "" {
			records, exists = historicalMapByMMSI[ship.MMSI]
		}

		// If no MMSI match, try ship name match
		if !exists || len(records) == 0 {
			normalizedShipName := normalizeShipName(ship.Name)
			if normalizedShipName != "" {
				records, exists = historicalMapByName[normalizedShipName]
			}
		}

		// Skip if no records found
		if !exists || len(records) == 0 {
			continue
		}

		// Extract historical trail (lat/lon pairs)
		trail := make([][]float64, 0, len(records))
		for _, record := range records {
			trail = append(trail, []float64{record.Latitude, record.Longitude})
		}
		ship.HistoricalTrail = trail

		// Use last record for current position if not set
		lastRecord := records[len(records)-1]
		if ship.Lat == 0 && ship.Lon == 0 {
			ship.Lat = lastRecord.Latitude
			ship.Lon = lastRecord.Longitude
		}

		// Use destination from last record if not set
		if ship.Destination == "" {
			ship.Destination = lastRecord.Destination
		}

		// Calculate ETA using ML service if destination exists
		if ship.Destination != "" && ship.Lat != 0 && ship.Lon != 0 {
			// Get destination coordinates
			destLat, destLon := getPortCoordinates(ship.Destination)
			if destLat != 0 && destLon != 0 {
				// Calculate distance to destination
				distance := haversineDistance(ship.Lat, ship.Lon, destLat, destLon)

				// Calculate historical average time for this route
				historicalAvg := calculateHistoricalAvgTime(ship.MMSI, ship.Destination)

				// Encode route ID from destination port
				routeID := encodeRouteID(ship.Destination)

				// Try to get ML prediction
				prediction, err := predictArrivalTime(distance, routeID, historicalAvg)
				if err == nil && prediction != nil {
					// Use ML predicted arrival time
					ship.ETA = prediction.EstimatedArrivalTime
				} else {
					// Fallback: use ETA from last record or calculate basic estimate
					if lastRecord.ETA != "" {
						ship.ETA = lastRecord.ETA
					} else if lastRecord.SpeedKnots > 0 {
						// Basic calculation: distance / speed
						// Convert knots to km/h (1 knot = 1.852 km/h)
						speedKmh := lastRecord.SpeedKnots * 1.852
						hoursToArrival := distance / speedKmh
						eta := time.Now().Add(time.Duration(hoursToArrival * float64(time.Hour)))
						ship.ETA = eta.Format(time.RFC3339)
					}
				}
			} else if lastRecord.ETA != "" {
				// Use historical ETA if destination coordinates not found
				ship.ETA = lastRecord.ETA
			}
		}
	}
}

// Port represents a port with coordinates and metadata
type Port struct {
	PortCode     string  `json:"port_code"`
	PortName     string  `json:"port_name"`
	Country      string  `json:"country"`
	Region       string  `json:"region"`
	Latitude     float64 `json:"latitude"`
	Longitude    float64 `json:"longitude"`
	PortType     string  `json:"port_type"`
	CapacityTons int     `json:"capacity_tons"`
	DraftMeters  float64 `json:"draft_meters"`
	Berths       int     `json:"berths"`
}

// PortsData contains all ports information
type PortsData struct {
	Metadata struct {
		Description      string `json:"description"`
		TotalPorts       int    `json:"total_ports"`
		LastUpdated      string `json:"last_updated"`
		CoordinateSystem string `json:"coordinate_system"`
	} `json:"metadata"`
	Ports []Port `json:"ports"`
}

// Global ports cache
var portsCache *PortsData

// loadPortsData loads ports from ports.json file
func loadPortsData(filepath string) (*PortsData, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var data PortsData
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, err
	}

	return &data, nil
}

// getPortsData returns cached ports data or loads from file
func getPortsData() *PortsData {
	if portsCache != nil {
		return portsCache
	}

	// Try to load ports data
	data, err := loadPortsData("backend/data/ports.json")
	if err != nil {
		data, err = loadPortsData("data/ports.json")
		if err != nil {
			// Return empty ports data if file not found
			return &PortsData{}
		}
	}

	portsCache = data
	return portsCache
}

// generatePredictedRoutes creates predicted route waypoints for ships with destinations
func generatePredictedRoutes(ships *[]ShipData, masterData *ShipMasterData) {
	for i := range *ships {
		ship := &(*ships)[i]

		// Skip if no current position or destination
		if ship.Lat == 0 || ship.Lon == 0 {
			continue
		}

		// Get destination coordinates from current_voyage in masterData
		var destLat, destLon float64

		// Find destination coordinates from master data
		for _, master := range masterData.BulkCarriers {
			if master.VesselMMSI == ship.MMSI && master.CurrentVoyage != nil {
				destLat, destLon = getPortCoordinates(master.CurrentVoyage.Destination)
				break
			}
		}

		// If destination coordinates not found, skip
		if destLat == 0 && destLon == 0 {
			continue
		}

		// Update ship destination coordinates
		ship.DestinationLat = destLat
		ship.DestinationLon = destLon

		// Generate simple route (3-5 waypoints from current to destination)
		route := generateSimpleRoute(ship.Lat, ship.Lon, destLat, destLon)
		ship.Route = route
		ship.CurrentRouteIndex = 0
	}
}

// getPortCoordinates returns lat/lon for port by code from ports.json
func getPortCoordinates(portCode string) (float64, float64) {
	portsData := getPortsData()

	// Normalize port code (uppercase, trim)
	normalizedCode := strings.ToUpper(strings.TrimSpace(portCode))

	// Search for port by code
	for _, port := range portsData.Ports {
		if port.PortCode == normalizedCode {
			return port.Latitude, port.Longitude
		}
	}

	// Port not found
	return 0, 0
}

// generateSimpleRoute creates waypoints from start to destination
func generateSimpleRoute(startLat, startLon, destLat, destLon float64) [][]float64 {
	route := [][]float64{
		{startLat, startLon}, // Current position
	}

	// Calculate number of intermediate waypoints (3-5 total points)
	numWaypoints := 3

	for i := 1; i <= numWaypoints; i++ {
		fraction := float64(i) / float64(numWaypoints+1)
		lat := startLat + (destLat-startLat)*fraction
		lon := startLon + (destLon-startLon)*fraction
		route = append(route, []float64{lat, lon})
	}

	// Add final destination
	route = append(route, []float64{destLat, destLon})

	return route
}
