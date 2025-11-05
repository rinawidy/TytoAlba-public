package storage

import (
	"sync"
	"time"

	"github.com/your-org/tytoalba/backend/internal/mqtt"
)

// ShipStore manages in-memory storage of ship data
type ShipStore struct {
	mu       sync.RWMutex
	ships    map[string]*mqtt.ShipData          // MMSI -> latest ship data
	history  map[string][]mqtt.ShipData         // MMSI -> historical data (last 24h)
	maxAge   time.Duration                       // Maximum age for historical data
}

// NewShipStore creates a new ship data store
func NewShipStore() *ShipStore {
	store := &ShipStore{
		ships:   make(map[string]*mqtt.ShipData),
		history: make(map[string][]mqtt.ShipData),
		maxAge:  24 * time.Hour, // Keep 24 hours of history
	}

	// Start cleanup goroutine
	go store.cleanupOldData()

	return store
}

// UpdateShipData updates or adds ship data
func (s *ShipStore) UpdateShipData(data mqtt.ShipData) {
	s.mu.Lock()
	defer s.mu.Unlock()

	mmsi := data.VesselMMSI

	// Update latest data
	s.ships[mmsi] = &data

	// Add to history
	s.history[mmsi] = append(s.history[mmsi], data)

	// Keep only last 100 records per ship in memory
	if len(s.history[mmsi]) > 100 {
		s.history[mmsi] = s.history[mmsi][len(s.history[mmsi])-100:]
	}
}

// GetShip returns the latest data for a specific ship
func (s *ShipStore) GetShip(mmsi string) (*mqtt.ShipData, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ship, exists := s.ships[mmsi]
	return ship, exists
}

// GetAllShips returns all ship data
func (s *ShipStore) GetAllShips() []mqtt.ShipData {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ships := make([]mqtt.ShipData, 0, len(s.ships))
	for _, ship := range s.ships {
		ships = append(ships, *ship)
	}

	return ships
}

// GetBulkCarriers returns only bulk carrier ships
func (s *ShipStore) GetBulkCarriers() []mqtt.ShipData {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ships := make([]mqtt.ShipData, 0)
	for _, ship := range s.ships {
		if ship.ShipType == "bulk_carrier" {
			ships = append(ships, *ship)
		}
	}

	return ships
}

// GetShipHistory returns historical data for a ship
func (s *ShipStore) GetShipHistory(mmsi string, hours int) []mqtt.ShipData {
	s.mu.RLock()
	defer s.mu.RUnlock()

	history, exists := s.history[mmsi]
	if !exists {
		return []mqtt.ShipData{}
	}

	// Filter by time
	cutoff := time.Now().UTC().Add(-time.Duration(hours) * time.Hour)
	filtered := make([]mqtt.ShipData, 0)

	for _, data := range history {
		if data.Timestamp.After(cutoff) {
			filtered = append(filtered, data)
		}
	}

	return filtered
}

// GetShipCount returns the total number of tracked ships
func (s *ShipStore) GetShipCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	return len(s.ships)
}

// GetShipCountByType returns ship counts by type
func (s *ShipStore) GetShipCountByType() map[string]int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	counts := make(map[string]int)
	for _, ship := range s.ships {
		counts[ship.ShipType]++
	}

	return counts
}

// cleanupOldData periodically removes old data
func (s *ShipStore) cleanupOldData() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		s.mu.Lock()
		cutoff := time.Now().UTC().Add(-s.maxAge)

		// Clean up history
		for mmsi, history := range s.history {
			filtered := make([]mqtt.ShipData, 0)
			for _, data := range history {
				if data.Timestamp.After(cutoff) {
					filtered = append(filtered, data)
				}
			}
			s.history[mmsi] = filtered
		}

		// Remove ships with no recent data
		for mmsi, ship := range s.ships {
			if ship.Timestamp.Before(cutoff) {
				delete(s.ships, mmsi)
				delete(s.history, mmsi)
			}
		}

		s.mu.Unlock()
	}
}
