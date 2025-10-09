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
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <p class="text-sm text-gray-600 mb-1">Selected Ship</p>
            <p class="text-lg font-semibold text-gray-900">{{ selectedShip?.name || 'N/A' }}</p>
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
            <p class="text-lg font-semibold text-blue-600">{{ selectedShip?.estimatedFuel || 0 }} L</p>
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
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Ship Name</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Port Destination</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">ETA</th>
                <th class="px-4 py-3 text-left text-sm font-semibold text-gray-900">Estimated Fuel on Arrival</th>
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
                <td class="px-4 py-3 text-sm text-gray-900 font-medium">{{ ship.name }}</td>
                <td class="px-4 py-3 text-sm text-gray-900">{{ ship.destination }}</td>
                <td class="px-4 py-3 text-sm text-gray-900">{{ formatETA(ship.eta) }}</td>
                <td class="px-4 py-3 text-sm text-gray-900">{{ ship.estimatedFuel.toLocaleString() }} L</td>
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

interface ShipData {
  id: string
  name: string
  lat: number
  lon: number
  destination: string
  eta: Date
  estimatedFuel: number
  status: string
  route: Array<[number, number]>
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
  if (map) {
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

// Initialize map
const initMap = () => {
  // Create map centered on Indonesia
  map = L.map('map').setView([-2.5, 118.0], 5)

  // Add tile layer (Light mode)
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(map)

  // Mock ship data
  ships.value = [
    {
      id: '1',
      name: 'Rasuna Baruna',
      lat: -6.906,
      lon: 110.831,
      destination: 'Taboneo Port',
      eta: new Date(Date.now() + 24 * 3600 * 1000),
      estimatedFuel: 8500,
      status: 'In Progress',
      route: [
        [-6.906, 110.831],  // Jepara
        [-5.5, 112.5],
        [-4.2, 114.8],
        [-2.8, 116.2],      // Taboneo
      ]
    },
    {
      id: '2',
      name: 'Latifah Baruna',
      lat: -2.8,
      lon: 116.2,
      destination: 'Labuan Bajo',
      eta: new Date(Date.now() + 36 * 3600 * 1000),
      estimatedFuel: 7200,
      status: 'In Progress',
      route: [
        [-2.8, 116.2],      // Taboneo
        [-4.0, 118.0],
        [-6.5, 120.5],
        [-8.497, 119.883],  // Labuan Bajo
      ]
    },
    {
      id: '3',
      name: 'Martha Baruna',
      lat: -6.906,
      lon: 110.831,
      destination: 'Jepara Port',
      eta: new Date(Date.now() + 2 * 3600 * 1000),
      estimatedFuel: 9800,
      status: 'In Port',
      route: [
        [-6.906, 110.831],
      ]
    },
    {
      id: '4',
      name: 'Meutia Baruna',
      lat: -1.5,
      lon: 117.5,
      destination: 'Makassar Port',
      eta: new Date(Date.now() + 48 * 3600 * 1000),
      estimatedFuel: 6500,
      status: 'In Progress',
      route: [
        [-0.5, 117.0],      // Kalimantan
        [-1.5, 117.5],
        [-3.0, 119.0],
        [-5.147, 119.432],  // Makassar
      ]
    },
  ]

  selectedShip.value = ships.value[0]

  // Add ships and routes to map
  ships.value.forEach((ship, index) => {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    const color = colors[index % colors.length]

    // Add route line (dotted)
    if (ship.route.length > 1) {
      const routeLine = L.polyline(ship.route, {
        color: color,
        weight: 3,
        opacity: 0.7,
        dashArray: '10, 10',
        dashOffset: '0'
      }).addTo(map!)
      routeLines.push(routeLine)
    }

    // Add ship marker
    const marker = L.marker([ship.lat, ship.lon], {
      icon: shipIcon(color)
    }).addTo(map!)

    marker.bindPopup(`
      <div class="p-2">
        <h3 class="font-bold text-gray-900">${ship.name}</h3>
        <p class="text-sm text-gray-600">Destination: ${ship.destination}</p>
        <p class="text-sm text-gray-600">ETA: ${formatETA(ship.eta)}</p>
        <p class="text-sm text-gray-600">Fuel Est.: ${ship.estimatedFuel.toLocaleString()} L</p>
      </div>
    `)

    shipMarkers.push(marker)
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
