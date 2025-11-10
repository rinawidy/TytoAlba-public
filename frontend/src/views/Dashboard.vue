<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <div class="bg-white border-b border-gray-200 px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <Ship class="w-8 h-8 text-blue-500" />
          <div>
            <h1 class="text-2xl font-bold text-gray-900">TytoAlba</h1>
            <p class="text-sm text-gray-600">Ship Route & Fuel Management</p>
          </div>
        </div>
        <div class="flex items-center gap-4">
          <div class="px-4 py-2 rounded-lg bg-white border border-gray-200">
            <span class="text-sm text-gray-600">{{ currentDateTime }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <div class="p-6">
      <!-- Arrival Prediction Info Panel -->
      <div class="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <div class="flex items-center gap-3 mb-4">
          <Clock class="w-6 h-6 text-green-500" />
          <h2 class="text-xl font-semibold text-gray-900">Arrival Prediction</h2>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-5 gap-4">
          <div>
            <p class="text-sm text-gray-600 mb-1">Selected Ship</p>
            <p class="text-lg font-semibold text-gray-900">{{ selectedShip?.name || 'N/A' }}</p>
            <p class="text-xs text-gray-500">{{ selectedShip?.type || '' }}</p>
          </div>
          <div>
            <p class="text-sm text-gray-600 mb-1">Coal Capacity</p>
            <p class="text-lg font-semibold text-orange-600">
              <span v-if="selectedShip?.type === 'Tugboat' && selectedShip?.bargeCoalCapacity">
                {{ selectedShip.bargeCoalCapacity?.toLocaleString() || 0 }} tons
              </span>
              <span v-else>
                {{ selectedShip?.coalCapacity?.toLocaleString() || 0 }} tons
              </span>
            </p>
          </div>
          <div>
            <p class="text-sm text-gray-600 mb-1">Estimated Arrival</p>
            <p class="text-lg font-semibold text-green-600">{{ formatETA(selectedShip?.eta) }}</p>
          </div>
          <div>
            <p class="text-sm text-gray-600 mb-1">Port Destination</p>
            <p class="text-lg font-semibold text-gray-900">{{ selectedShip?.destination || 'N/A' }}</p>
          </div>
          <div>
            <p class="text-sm text-gray-600 mb-1">Fuel on Arrival (Est.)</p>
            <p class="text-lg font-semibold text-blue-600">{{ selectedShip?.estimatedFuel?.toLocaleString() || 'N/A' }} L</p>
          </div>
        </div>
      </div>

      <!-- Map Section -->
      <div class="bg-white rounded-lg border border-gray-200 p-6 mb-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Ship Route Map - Indonesia</h3>
        <div id="map" class="w-full h-[600px] rounded-lg overflow-hidden"></div>
      </div>

      <!-- Ship Information Table -->
      <div class="bg-white rounded-lg border border-gray-200 p-6">
        <h3 class="text-lg font-semibold text-gray-900 mb-4">Ship Information</h3>
        <div class="overflow-x-auto">
          <table class="w-full">
            <thead class="bg-gray-100">
              <tr>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">No</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">MMSI</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Ship Name</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Type</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Coal Capacity (tons)</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Port Destination</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">ETA</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(ship, idx) in ships"
                :key="ship.id"
                class="border-b border-gray-200 hover:bg-gray-50 cursor-pointer"
                @click="selectShip(ship)"
                :class="{ 'bg-blue-50': selectedShip?.id === ship.id }"
              >
                <td class="px-4 py-3 text-sm text-gray-900">{{ idx + 1 }}</td>
                <td class="px-4 py-3 text-sm text-gray-600 font-mono">{{ ship.mmsi }}</td>
                <td class="px-4 py-3 text-sm text-gray-900 font-medium">{{ ship.name }}</td>
                <td class="px-4 py-3 text-sm text-gray-900">{{ ship.type }}</td>
                <td class="px-4 py-3 text-sm text-gray-900">
                  <span v-if="ship.type === 'Tugboat' && ship.bargeCoalCapacity">
                    {{ ship.bargeCoalCapacity.toLocaleString() }} (barge: {{ ship.pushingBarge }})
                  </span>
                  <span v-else-if="ship.type === 'Barge' && ship.pushedBy">
                    {{ ship.coalCapacity.toLocaleString() }} (by: {{ ship.pushedBy }})
                  </span>
                  <span v-else>
                    {{ ship.coalCapacity.toLocaleString() }}
                  </span>
                </td>
                <td class="px-4 py-3 text-sm text-gray-900">{{ ship.destination || 'N/A' }}</td>
                <td class="px-4 py-3 text-sm text-gray-900">{{ formatETA(ship.eta) }}</td>
                <td class="px-4 py-3 text-sm">
                  <span
                    class="px-2 py-1 text-xs rounded"
                    :class="getStatusClass(ship.status)"
                  >
                    {{ ship.status }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { Ship, Clock } from 'lucide-vue-next'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import shipsData from '../data/ships.json'

interface ShipData {
  id: string
  mmsi: string
  name: string
  type: string  // "Bulk Carrier", "Tugboat", "Barge"
  coalCapacity: number
  loa: number
  beam: number
  dwt: number
  status: string
  pushingBarge?: string  // For tugboats
  bargeCoalCapacity?: number  // Coal capacity of barge being pushed
  pushedBy?: string  // For barges
  // Map display fields (optional for tugboats/barges)
  lat?: number
  lon?: number
  destination?: string
  eta?: Date
  estimatedFuel?: number
  route?: Array<[number, number]>
  currentRouteIndex?: number
}

const currentDateTime = ref('')
const selectedShip = ref<ShipData | null>(null)
const ships = ref<ShipData[]>([])
let map: L.Map | null = null
let shipMarkers: L.Marker[] = []
let routeLines: L.Polyline[] = []

// Update current date/time
const updateDateTime = () => {
  const now = new Date()
  currentDateTime.value = now.toLocaleString('id-ID', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZone: 'Asia/Jakarta'
  }) + ' WIB'
}

// Format ETA
const formatETA = (eta: Date | undefined) => {
  if (!eta) return 'N/A'
  return new Date(eta).toLocaleString('id-ID', {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// Get status class
const getStatusClass = (status: string) => {
  switch (status.toLowerCase()) {
    case 'in progress':
      return 'bg-blue-100 text-blue-700'
    case 'in port':
      return 'bg-green-100 text-green-700'
    case 'delayed':
      return 'bg-red-100 text-red-700'
    default:
      return 'bg-gray-100 text-gray-700'
  }
}

// Select ship
const selectShip = (ship: ShipData) => {
  selectedShip.value = ship
  if (map && ship.lat !== undefined && ship.lon !== undefined) {
    map.setView([ship.lat, ship.lon], 7)
  }
}

// Ship icon HTML
const shipIcon = (color: string) => {
  return L.divIcon({
    html: `
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M3 13.5L12 3L21 13.5" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M3 13.5L5 21H19L21 13.5" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="12" cy="12" r="10" fill="${color}" fill-opacity="0.2"/>
      </svg>
    `,
    className: 'ship-marker',
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  })
}

// Fetch ships from backend
const fetchShips = async () => {
  try {
    const response = await fetch('http://localhost:8080/api/ships')
    if (!response.ok) {
      throw new Error('Failed to fetch ships')
    }
    const data = await response.json()
    ships.value = data
    if (ships.value.length > 0) {
      selectedShip.value = ships.value[0]
    }
  } catch (error) {
    console.error('Error fetching ships:', error)
    // Fallback to mock data if backend is not available
    loadMockData()
  }
}

// Load mock data (fallback)
const loadMockData = () => {
  // Load ships from ships.json
  const loadedShips = shipsData.ships.map((ship: any) => {
    // Add mock position and route data for bulk carriers (for map display)
    if (ship.type === 'Bulk Carrier') {
      const destinations = ['Taboneo Port', 'Labuan Bajo', 'Makassar Port', 'Jepara Port', 'Surabaya Port']
      const routes = [
        [[-6.906, 110.831], [-5.5, 112.5], [-4.2, 114.8], [-2.8, 116.2]],  // To Taboneo
        [[-2.8, 116.2], [-4.0, 118.0], [-6.5, 120.5], [-8.497, 119.883]],  // To Labuan Bajo
        [[-0.5, 117.0], [-1.5, 117.5], [-3.0, 119.0], [-5.147, 119.432]],  // To Makassar
        [[-6.906, 110.831]],  // In Jepara Port
        [[-7.250, 112.750], [-7.200, 112.700], [-7.180, 112.740]]  // To Surabaya
      ]

      const routeIdx = parseInt(ship.id) % routes.length
      const route = routes[routeIdx]
      const destination = destinations[routeIdx]

      return {
        ...ship,
        lat: route[Math.min(1, route.length - 1)][0],
        lon: route[Math.min(1, route.length - 1)][1],
        destination: destination,
        eta: new Date(Date.now() + (12 + parseInt(ship.id) * 6) * 3600 * 1000),
        estimatedFuel: 6500 + parseInt(ship.id) * 200,
        route: route,
        currentRouteIndex: Math.min(1, route.length - 1)
      }
    }
    // For tugboats and barges, no route data needed
    return ship
  })

  ships.value = loadedShips
  if (ships.value.length > 0) {
    // Select first bulk carrier for initial display
    selectedShip.value = ships.value.find(s => s.type === 'Bulk Carrier') || ships.value[0]
  }
}

// Initialize map
const initMap = async () => {
  // Create map centered on Indonesia
  map = L.map('map').setView([-2.5, 118.0], 5)

  // Add tile layer (Light mode)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map)

  // Fetch ship data from backend
  await fetchShips()

  // Add ships and routes to map (only bulk carriers with route data)
  const bulkCarriers = ships.value.filter(ship => ship.route && ship.route.length > 0)

  bulkCarriers.forEach((ship, index) => {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
    const color = colors[index % colors.length]

    // Add route lines with solid (traversed) and dotted (remaining) segments
    if (ship.route && ship.route.length > 1) {
      const currentIdx = ship.currentRouteIndex || 0

      // Traversed route (solid line, green)
      if (currentIdx > 0) {
        const traversedRoute = ship.route.slice(0, currentIdx + 1)
        const traversedLine = L.polyline(traversedRoute, {
          color: '#10b981',  // Green color for completed path
          weight: 4,
          opacity: 0.8,
          dashArray: '',  // Solid line
        }).addTo(map!)
        routeLines.push(traversedLine)
      }

      // Remaining route (dotted line, blue/gray)
      if (currentIdx < ship.route.length - 1) {
        const remainingRoute = ship.route.slice(currentIdx)
        const remainingLine = L.polyline(remainingRoute, {
          color: '#94a3b8',  // Gray color for remaining path
          weight: 3,
          opacity: 0.6,
          dashArray: '10, 10',  // Dotted line
          dashOffset: '0'
        }).addTo(map!)
        routeLines.push(remainingLine)
      }
    }

    // Add ship marker (only if lat/lon exists)
    if (ship.lat !== undefined && ship.lon !== undefined) {
      const marker = L.marker([ship.lat, ship.lon], {
        icon: shipIcon(color)
      }).addTo(map!)

      marker.bindPopup(`
        <div class="p-2">
          <h3 class="font-bold text-gray-900">${ship.name}</h3>
          <p class="text-sm text-gray-600">Type: ${ship.type}</p>
          <p class="text-sm text-gray-600">Coal Capacity: ${ship.coalCapacity.toLocaleString()} tons</p>
          <p class="text-sm text-gray-600">Destination: ${ship.destination || 'N/A'}</p>
          <p class="text-sm text-gray-600">ETA: ${formatETA(ship.eta)}</p>
          <p class="text-sm text-gray-600">Fuel Est.: ${ship.estimatedFuel?.toLocaleString() || 'N/A'} L</p>
        </div>
      `)

      shipMarkers.push(marker)
    }
  })
}

// Lifecycle
onMounted(() => {
  updateDateTime()
  setInterval(updateDateTime, 1000)

  // Initialize map after a short delay to ensure DOM is ready
  setTimeout(initMap, 100)
})

onUnmounted(() => {
  if (map) {
    map.remove()
    map = null
  }
})
</script>

<style scoped>
#map {
  height: 600px;
  width: 100%;
  z-index: 1;
}

:deep(.ship-marker) {
  background: transparent;
  border: none;
}

:deep(.leaflet-popup-content-wrapper) {
  border-radius: 8px;
}
</style>
