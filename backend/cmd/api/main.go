package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/your-org/tytoalba/backend/internal/handlers"
	"github.com/your-org/tytoalba/backend/internal/mqtt"
	"github.com/your-org/tytoalba/backend/internal/storage"
)

var (
	mqttBroker *mqtt.MQTTBroker
	shipStore  *storage.ShipStore
)

func main() {
	fmt.Println("=" + string(make([]byte, 68)) + "=")
	fmt.Println("  TytoAlba Backend Server with MQTT Support")
	fmt.Println("=" + string(make([]byte, 68)) + "=")

	// Initialize ship store
	shipStore = storage.NewShipStore()
	handlers.ShipStore = shipStore
	log.Println("✓ Ship data store initialized")

	// Initialize and connect to MQTT broker
	brokerURL := getEnv("MQTT_BROKER_URL", "tcp://localhost:1883")
	clientID := getEnv("MQTT_CLIENT_ID", "tytoalba-backend")
	username := getEnv("MQTT_USERNAME", "")
	password := getEnv("MQTT_PASSWORD", "")

	mqttBroker = mqtt.NewMQTTBroker(brokerURL, clientID, username, password)

	// Set data handler to update ship store
	mqttBroker.SetDataHandler(func(data mqtt.ShipData) {
		shipStore.UpdateShipData(data)
		log.Printf("✓ Updated ship data: MMSI=%s, Type=%s", data.VesselMMSI, data.ShipType)
	})

	// Connect to MQTT broker
	if err := mqttBroker.Connect(); err != nil {
		log.Printf("⚠️  Warning: MQTT connection failed: %v", err)
		log.Println("   Server will continue without MQTT (fallback to file-based data)")
	} else {
		log.Println("✓ MQTT broker connected successfully")
	}

	// Register routes
	http.HandleFunc("/api/ships", handlers.GetShips)                    // Legacy file-based endpoint
	http.HandleFunc("/api/mqtt/ships", handlers.GetShipsFromMQTT)       // MQTT-based all ships
	http.HandleFunc("/api/mqtt/bulk-carriers", handlers.GetBulkCarriers) // Only bulk carriers
	http.HandleFunc("/api/mqtt/ship", handlers.GetShipByMMSI)           // Single ship by MMSI
	http.HandleFunc("/api/mqtt/history", handlers.GetShipHistory)       // Ship history
	http.HandleFunc("/api/mqtt/stats", handlers.GetShipStats)           // Ship statistics
	http.HandleFunc("/api/historical/voyages", handlers.GetHistoricalVoyages)  // All historical data
	http.HandleFunc("/api/historical/vessel", handlers.GetVesselHistory)        // Historical data for specific vessel
	http.HandleFunc("/api/historical/stats", handlers.GetHistoricalStats)       // Historical data statistics
	http.HandleFunc("/api/historical/recent", handlers.GetRecentPositions)      // Recent positions
	http.HandleFunc("/health", healthCheck)
	http.HandleFunc("/mqtt/status", mqttStatus)

	// Start server
	port := getEnv("PORT", ":8080")
	fmt.Printf("\n🚀 Server starting on port %s\n\n", port)
	fmt.Println("REST API Endpoints:")
	fmt.Println("  - GET  /health                   - Health check")
	fmt.Println("  - GET  /mqtt/status              - MQTT broker status")
	fmt.Println("  - GET  /api/ships                - Get ships (file-based, legacy)")
	fmt.Println("  - GET  /api/mqtt/ships           - Get all ships (MQTT)")
	fmt.Println("  - GET  /api/mqtt/bulk-carriers   - Get bulk carriers only")
	fmt.Println("  - GET  /api/mqtt/ship?mmsi=XXX   - Get ship by MMSI")
	fmt.Println("  - GET  /api/mqtt/history?mmsi=XXX&hours=24 - Get ship history")
	fmt.Println("  - GET  /api/mqtt/stats           - Get ship statistics")
	fmt.Println("\nHistorical Data Endpoints:")
	fmt.Println("  - GET  /api/historical/voyages   - Get all historical voyages")
	fmt.Println("  - GET  /api/historical/vessel?mmsi=XXX - Get vessel history by MMSI")
	fmt.Println("  - GET  /api/historical/stats     - Get historical data statistics")
	fmt.Println("  - GET  /api/historical/recent?hours=24 - Get recent positions")
	fmt.Println("\nMQTT Topics (subscribing):")
	fmt.Println("  - tytoalba/ships/+/ais           - AIS position data")
	fmt.Println("  - tytoalba/ships/+/sensors       - Sensor data (fuel, engine)")
	fmt.Println("  - tytoalba/ships/+/status        - Status updates")
	fmt.Println()

	if err := http.ListenAndServe(port, nil); err != nil {
		log.Fatal(err)
	}
}

func healthCheck(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)

	mqttConnected := false
	if mqttBroker != nil {
		mqttConnected = mqttBroker.IsConnected()
	}

	shipCount := 0
	if shipStore != nil {
		shipCount = shipStore.GetShipCount()
	}

	response := map[string]interface{}{
		"status":         "healthy",
		"service":        "tytoalba-backend",
		"mqtt_connected": mqttConnected,
		"ships_tracked":  shipCount,
	}

	json.NewEncoder(w).Encode(response)
}

func mqttStatus(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")

	if mqttBroker == nil {
		http.Error(w, `{"error":"MQTT broker not initialized"}`, http.StatusServiceUnavailable)
		return
	}

	stats := mqttBroker.GetStats()
	stats["ships_tracked"] = shipStore.GetShipCount()
	stats["ships_by_type"] = shipStore.GetShipCountByType()

	json.NewEncoder(w).Encode(stats)
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
