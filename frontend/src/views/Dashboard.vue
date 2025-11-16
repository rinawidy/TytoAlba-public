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
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold text-gray-900">Ship Route Map - Indonesia</h3>

          <!-- Weather Layer Controls -->
          <div class="flex gap-2">
            <button
              v-if="!isWindyAvailable"
              disabled
              class="px-4 py-2 rounded-lg text-sm font-medium bg-gray-200 text-gray-500 cursor-not-allowed"
            >
              Weather (Windy unavailable)
            </button>
            <template v-else>
              <button
                @click="toggleWeatherLayer('wind')"
                :class="[
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  activeWeatherLayer === 'wind'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                ]"
              >
                Wind
              </button>
              <button
                @click="toggleWeatherLayer('waves')"
                :class="[
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  activeWeatherLayer === 'waves'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                ]"
              >
                Waves
              </button>
              <button
                @click="toggleWeatherLayer('temp')"
                :class="[
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  activeWeatherLayer === 'temp'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                ]"
              >
                Temperature
              </button>
              <button
                @click="toggleWeatherLayer('none')"
                :class="[
                  'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                  activeWeatherLayer === 'none'
                    ? 'bg-gray-500 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                ]"
              >
                Hide Weather
              </button>
            </template>
          </div>
        </div>
        <div id="windy" class="w-full h-[600px] rounded-lg overflow-hidden"></div>
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
                v-for="(ship, idx) in shipsWithData"
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
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { Ship, Clock } from 'lucide-vue-next'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Declare Windy API types
declare global {
  interface Window {
    windyInit: any
  }
}

const WINDY_API_KEY = import.meta.env.VITE_WINDY_API_KEY || '8izAhGv0nYV492ergPznqLcrGr4N6r0r'

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
  route?: Array<[number, number]>  // Predicted future route (dashed line)
  currentRouteIndex?: number
  historicalTrail?: Array<[number, number]>  // Past positions (solid line)
}

const currentDateTime = ref('')
const selectedShip = ref<ShipData | null>(null)
const ships = ref<ShipData[]>([])
const activeWeatherLayer = ref<'wind' | 'waves' | 'temp' | 'none'>('none')
const isWindyAvailable = ref(false)
let map: L.Map | null = null
let windyMap: any = null
let shipMarkers: L.Marker[] = []
let routeLines: L.Polyline[] = []

// Computed: Filter ships with position data for table display
const shipsWithData = computed(() => {
  return ships.value.filter(ship =>
    ship.lat !== undefined &&
    ship.lon !== undefined &&
    ship.lat !== 0 &&
    ship.lon !== 0
  )
})

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
  return L.icon({
    iconUrl: new URL('../components/870107.png', import.meta.url).href,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
    popupAnchor: [0, -16],
    className: 'ship-marker'
  })
}

// Fetch ships from backend (only source of truth is ships_master.json via API)
const fetchShips = async () => {
  try {
    const response = await fetch('http://localhost:8080/api/ships')
    if (!response.ok) {
      throw new Error('Failed to fetch ships from backend')
    }
    const data = await response.json()

    ships.value = data
    console.log('Ships loaded from API:', ships.value.length)
    console.log('Ships with position data:', shipsWithData.value.length)

    // Log first ship for debugging
    if (ships.value.length > 0) {
      console.log('First ship:', ships.value[0].name)
      console.log('First ship trail:', ships.value[0].historicalTrail?.length || 0, 'points')
      console.log('First ship route:', ships.value[0].route?.length || 0, 'waypoints')
    }

    // Select first ship with data
    if (shipsWithData.value.length > 0) {
      selectedShip.value = shipsWithData.value[0]
    }
  } catch (error) {
    console.error('Error fetching ships from backend:', error)
    console.error('Backend API not available. Please ensure backend server is running.')
    ships.value = []
  }
}

// Toggle weather layer
const toggleWeatherLayer = (layer: 'wind' | 'waves' | 'temp' | 'none') => {
  activeWeatherLayer.value = layer

  if (windyMap && windyMap.store) {
    const overlayMap: Record<string, string> = {
      'wind': 'wind',
      'waves': 'waves',
      'temp': 'temp',
      'none': 'wind' // Default to wind but we'll handle visibility
    }

    const overlay = overlayMap[layer]
    windyMap.store.set('overlay', overlay)

    // Toggle particle animation visibility
    if (layer === 'none') {
      windyMap.store.set('particlesAnim', false)
    } else {
      windyMap.store.set('particlesAnim', true)
    }
  }
}

// Wait for Windy API to be ready (with timeout)
const waitForWindy = () => {
  return new Promise<boolean>((resolve) => {
    if (typeof window.windyInit !== 'undefined') {
      resolve(true)
      return
    }

    let attempts = 0
    const maxAttempts = 30 // 3 seconds timeout

    const checkWindy = setInterval(() => {
      attempts++
      if (typeof window.windyInit !== 'undefined') {
        clearInterval(checkWindy)
        resolve(true)
      } else if (attempts >= maxAttempts) {
        clearInterval(checkWindy)
        console.warn('Windy API failed to load, falling back to Leaflet')
        resolve(false)
      }
    }, 100)
  })
}

// Add ship markers and routes to map
const addShipsToMap = () => {
  // Add ships and routes to map (ships with position data)
  const bulkCarriers = ships.value.filter(ship => ship.lat !== undefined && ship.lon !== undefined)

  bulkCarriers.forEach((ship, index) => {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316']
    const color = colors[index % colors.length]

    // 1. Draw HISTORICAL TRAIL (solid green line) - from historicalTrail data
    if (ship.historicalTrail && ship.historicalTrail.length > 1) {
      const historicalLine = L.polyline(ship.historicalTrail, {
        color: '#10b981',  // Green color for historical path
        weight: 4,
        opacity: 0.8,
        dashArray: '',  // Solid line
      }).addTo(map!)

      historicalLine.bindPopup(`
        <div class="p-2">
          <h4 class="font-bold text-green-600">Historical Trail</h4>
          <p class="text-sm text-gray-600">${ship.name}</p>
          <p class="text-xs text-gray-500">${ship.historicalTrail.length} positions recorded</p>
        </div>
      `)

      routeLines.push(historicalLine)
    }

    // 2. Draw PREDICTED ROUTE (dashed blue/gray line) - from route data
    if (ship.route && ship.route.length > 1) {
      const predictedLine = L.polyline(ship.route, {
        color: '#3b82f6',  // Blue color for predicted route
        weight: 3,
        opacity: 0.6,
        dashArray: '10, 10',  // Dashed line
        dashOffset: '0'
      }).addTo(map!)

      predictedLine.bindPopup(`
        <div class="p-2">
          <h4 class="font-bold text-blue-600">Predicted Route</h4>
          <p class="text-sm text-gray-600">To: ${ship.destination || 'Unknown'}</p>
          <p class="text-xs text-gray-500">ETA: ${formatETA(ship.eta)}</p>
        </div>
      `)

      routeLines.push(predictedLine)
    }

    // 3. Add SHIP MARKER at current position
    if (ship.lat !== undefined && ship.lon !== undefined) {
      const marker = L.marker([ship.lat, ship.lon], {
        icon: shipIcon(color || '#3b82f6')
      }).addTo(map!)

      marker.bindPopup(`
        <div class="p-2">
          <h3 class="font-bold text-gray-900">${ship.name || 'Unknown'}</h3>
          <p class="text-sm text-gray-600">MMSI: ${ship.mmsi || 'N/A'}</p>
          <p class="text-sm text-gray-600">Type: ${ship.type || 'N/A'}</p>
          <p class="text-sm text-gray-600">Coal Capacity: ${ship.coalCapacity?.toLocaleString() || 'N/A'} tons</p>
          <p class="text-sm text-gray-600">Destination: ${ship.destination || 'N/A'}</p>
          <p class="text-sm text-gray-600">ETA: ${formatETA(ship.eta)}</p>
          <p class="text-sm text-gray-600">Fuel Est.: ${ship.estimatedFuel?.toLocaleString() || 'N/A'} L</p>
          <p class="text-xs text-gray-500 mt-2">
            <span class="inline-block w-3 h-0.5 bg-green-500 mr-1"></span> Historical Trail
            <span class="inline-block w-3 h-0.5 border-t-2 border-dashed border-blue-500 ml-3 mr-1"></span> Predicted Route
          </p>
        </div>
      `)

      shipMarkers.push(marker)
    }
  })
}

// Initialize Leaflet fallback map
const initLeafletMap = () => {
  console.log('Initializing Leaflet fallback map')

  // Create map centered on Indonesia
  map = L.map('windy').setView([-2.5, 118.0], 5)

  // Add tile layer (OpenStreetMap)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map)

  // Add ships to map
  addShipsToMap()
}

// Initialize Windy map
const initMap = async () => {
  try {
    // Fetch ship data from backend first
    await fetchShips()

    // Wait for Windy API to be available
    const windyLoaded = await waitForWindy()

    if (!windyLoaded) {
      // Fallback to Leaflet if Windy fails
      isWindyAvailable.value = false
      initLeafletMap()
      return
    }

    isWindyAvailable.value = true
    console.log('Windy API Key:', WINDY_API_KEY)

    // Initialize Windy Map
    const options = {
      key: WINDY_API_KEY,
      lat: -2.5,
      lon: 118.0,
      zoom: 5,
    }

    window.windyInit(options, (windyAPI: any) => {
      console.log('Windy initialized successfully')
      windyMap = windyAPI
      const { map: leafletMap } = windyAPI
      map = leafletMap

      // Add ships to map
      addShipsToMap()

      // Set initial overlay to none (hide weather by default)
      windyMap.store.set('overlay', 'wind')
      windyMap.store.set('particlesAnim', false)
    })
  } catch (error) {
    console.error('Error initializing map, falling back to Leaflet:', error)
    isWindyAvailable.value = false
    initLeafletMap()
  }
}

// Lifecycle
onMounted(() => {
  updateDateTime()
  setInterval(updateDateTime, 1000)

  // Initialize map after DOM is ready
  setTimeout(() => {
    initMap().catch(error => {
      console.error('Error initializing map:', error)
    })
  }, 500)
})

onUnmounted(() => {
  if (map) {
    map.remove()
    map = null
  }
})
</script>

<style scoped>
#windy {
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

:deep(.leaflet-control-container) {
  z-index: 10;
}
</style>
