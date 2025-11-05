package mqtt

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	mqtt "github.com/eclipse/paho.mqtt.golang"
)

// ShipData represents AIS data received from ships via MQTT
type ShipData struct {
	// Identification
	VesselMMSI   string    `json:"vessel_mmsi"`
	VesselName   string    `json:"vessel_name"`
	IMONumber    string    `json:"imo_number"`
	CallSign     string    `json:"call_sign"`
	ShipType     string    `json:"ship_type"` // "bulk_carrier", "tugboat", "barge"
	VesselType   string    `json:"vessel_type"` // More specific: "Bulk Carrier", "Tugboat", "Barge"

	// Dimensions (meters)
	LOA          float64   `json:"loa"`           // Length Overall
	Beam         float64   `json:"beam"`          // Width
	Draft        float64   `json:"draft"`         // Design draft
	Draught      float64   `json:"draught"`       // Current draught

	// Capacity & Tonnage
	GrossTonnage float64   `json:"gross_tonnage"` // GT
	DeadweightTonnage float64 `json:"deadweight_tonnage"` // DWT
	CoalCapacity float64   `json:"coal_capacity"` // Coal capacity (tons)

	// Technical Specifications
	EnginePower  float64   `json:"engine_power"`  // kW
	MaxSpeed     float64   `json:"max_speed"`     // knots
	FuelCapacity float64   `json:"fuel_capacity"` // liters
	BuildYear    int       `json:"build_year"`
	Flag         string    `json:"flag"`

	// Operational Data (from AIS/sensors)
	Timestamp    time.Time `json:"timestamp"`
	Latitude     float64   `json:"latitude"`
	Longitude    float64   `json:"longitude"`
	Speed        float64   `json:"speed"`         // Speed over ground (knots)
	Course       float64   `json:"course"`        // Course over ground (degrees)
	Heading      float64   `json:"heading"`       // True heading (degrees)
	Status       string    `json:"status"`        // Navigational status
	Destination  string    `json:"destination"`   // Destination port
	ETA          string    `json:"eta"`           // Estimated time of arrival
	FuelRemain   float64   `json:"fuel_remain"`   // Remaining fuel (liters)
	EngineRPM    float64   `json:"engine_rpm"`    // Engine RPM
	FuelRate     float64   `json:"fuel_rate"`     // Fuel consumption rate (L/hr)

	// For tugboat-barge combinations
	IsPushing    bool      `json:"is_pushing"`    // Is this tugboat pushing a barge
	BargeMMSI    string    `json:"barge_mmsi"`    // MMSI of barge being pushed
	TugboatMMSI  string    `json:"tugboat_mmsi"`  // MMSI of tugboat (if this is a barge)
}

// MQTTBroker handles MQTT connections and ship data
type MQTTBroker struct {
	client        mqtt.Client
	brokerURL     string
	clientID      string
	username      string
	password      string
	dataHandler   func(ShipData)
	connected     bool
}

// NewMQTTBroker creates a new MQTT broker instance
func NewMQTTBroker(brokerURL, clientID, username, password string) *MQTTBroker {
	return &MQTTBroker{
		brokerURL: brokerURL,
		clientID:  clientID,
		username:  username,
		password:  password,
		connected: false,
	}
}

// SetDataHandler sets the callback function for received ship data
func (b *MQTTBroker) SetDataHandler(handler func(ShipData)) {
	b.dataHandler = handler
}

// Connect establishes connection to MQTT broker
func (b *MQTTBroker) Connect() error {
	opts := mqtt.NewClientOptions()
	opts.AddBroker(b.brokerURL)
	opts.SetClientID(b.clientID)
	opts.SetUsername(b.username)
	opts.SetPassword(b.password)
	opts.SetCleanSession(true)
	opts.SetAutoReconnect(true)
	opts.SetConnectRetry(true)
	opts.SetConnectRetryInterval(10 * time.Second)
	opts.SetMaxReconnectInterval(5 * time.Minute)
	opts.SetKeepAlive(60 * time.Second)

	// Connection lost handler
	opts.SetConnectionLostHandler(func(client mqtt.Client, err error) {
		log.Printf("⚠️  MQTT Connection lost: %v", err)
		b.connected = false
	})

	// On connect handler
	opts.SetOnConnectHandler(func(client mqtt.Client) {
		log.Println("✓ MQTT Connected to broker")
		b.connected = true

		// Subscribe to ship data topics
		b.subscribeToTopics()
	})

	b.client = mqtt.NewClient(opts)

	log.Printf("🔌 Connecting to MQTT broker: %s", b.brokerURL)
	token := b.client.Connect()
	token.Wait()

	if err := token.Error(); err != nil {
		return fmt.Errorf("failed to connect to MQTT broker: %v", err)
	}

	return nil
}

// subscribeToTopics subscribes to all ship data topics
func (b *MQTTBroker) subscribeToTopics() {
	topics := map[string]byte{
		"tytoalba/ships/+/ais":     1, // AIS position data from all ships
		"tytoalba/ships/+/sensors": 1, // Sensor data (fuel, engine, etc.)
		"tytoalba/ships/+/status":  1, // Status updates
	}

	for topic, qos := range topics {
		token := b.client.Subscribe(topic, qos, b.messageHandler)
		token.Wait()

		if err := token.Error(); err != nil {
			log.Printf("❌ Failed to subscribe to %s: %v", topic, err)
		} else {
			log.Printf("✓ Subscribed to topic: %s (QoS %d)", topic, qos)
		}
	}
}

// messageHandler handles incoming MQTT messages
func (b *MQTTBroker) messageHandler(client mqtt.Client, msg mqtt.Message) {
	topic := msg.Topic()
	payload := msg.Payload()

	log.Printf("📨 Received message on topic: %s", topic)

	// Parse ship data from JSON
	var shipData ShipData
	if err := json.Unmarshal(payload, &shipData); err != nil {
		log.Printf("❌ Failed to parse ship data: %v", err)
		return
	}

	// Validate ship type (only bulk_carrier and pusher are accepted)
	if shipData.ShipType != "bulk_carrier" && shipData.ShipType != "pusher" {
		log.Printf("⚠️  Unknown ship type '%s' for MMSI %s, defaulting to bulk_carrier",
			shipData.ShipType, shipData.VesselMMSI)
		shipData.ShipType = "bulk_carrier"
	}

	// Set timestamp if not provided
	if shipData.Timestamp.IsZero() {
		shipData.Timestamp = time.Now().UTC()
	}

	// Log received data
	log.Printf("📍 Ship Data - MMSI: %s, Type: %s, Position: (%.4f, %.4f), Speed: %.1f kn",
		shipData.VesselMMSI, shipData.ShipType, shipData.Latitude, shipData.Longitude, shipData.Speed)

	// Call data handler if set
	if b.dataHandler != nil {
		b.dataHandler(shipData)
	}
}

// PublishCommand publishes a command to a specific ship
func (b *MQTTBroker) PublishCommand(vesselMMSI string, command interface{}) error {
	topic := fmt.Sprintf("tytoalba/ships/%s/commands", vesselMMSI)

	payload, err := json.Marshal(command)
	if err != nil {
		return fmt.Errorf("failed to marshal command: %v", err)
	}

	token := b.client.Publish(topic, 1, false, payload)
	token.Wait()

	if err := token.Error(); err != nil {
		return fmt.Errorf("failed to publish command: %v", err)
	}

	log.Printf("✓ Published command to %s", topic)
	return nil
}

// IsConnected returns true if connected to MQTT broker
func (b *MQTTBroker) IsConnected() bool {
	return b.connected && b.client.IsConnected()
}

// Disconnect disconnects from MQTT broker
func (b *MQTTBroker) Disconnect() {
	if b.client != nil && b.client.IsConnected() {
		b.client.Disconnect(250)
		log.Println("✓ Disconnected from MQTT broker")
	}
}

// GetStats returns MQTT broker statistics
func (b *MQTTBroker) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"connected":   b.connected,
		"broker_url":  b.brokerURL,
		"client_id":   b.clientID,
		"is_online":   b.client.IsConnected(),
	}
}
