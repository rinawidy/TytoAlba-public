import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Ship, Fuel, Navigation, Clock, TrendingUp, Sun, Moon, Globe, MapPin } from 'lucide-react';

const TytoAlbaDashboard = () => {
  const [theme, setTheme] = useState('dark');
  const [language, setLanguage] = useState('en');
  const [selectedShip, setSelectedShip] = useState(null);
  const [timeRange, setTimeRange] = useState('weekly');
  const [ships, setShips] = useState([]);
  const [voyageData, setVoyageData] = useState([]);
  const [fuelPrediction, setFuelPrediction] = useState(null);
  const [arrivalPrediction, setArrivalPrediction] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [activeShipLayer, setActiveShipLayer] = useState({});
  const itemsPerPage = 5;

  const translations = {
    en: {
      dashboard: 'Dashboard',
      fuelManagement: 'Fuel Management System',
      ships: 'Ships',
      fuelUsage: 'BBM Usage Diagram',
      mapTitle: 'Ship Map',
      detailUsage: 'Detail BBM Usage',
      shipUsageTable: 'Ship Usage Detail',
      fuelPrediction: 'Fuel Consumption Prediction',
      arrivalPrediction: 'Estimated Time of Arrival',
      predicted: 'Predicted',
      confidence: 'Confidence',
      remaining: 'Remaining Distance',
      daily: 'Daily',
      weekly: 'Weekly',
      monthly: 'Monthly',
      search: 'Search',
      showing: 'Showing',
      of: 'of',
      entries: 'entries',
      shipName: 'Ship Name',
      tank: 'Tank',
      volume: 'Volume',
      usage: 'Usage',
      datetime: 'Date & Time',
      status: 'Status',
      liters: 'liters',
      hours: 'hours',
      nm: 'NM',
      knots: 'knots',
      vessel: 'Vessel',
    },
    id: {
      dashboard: 'Dasbor',
      fuelManagement: 'Sistem Manajemen Bahan Bakar',
      ships: 'Kapal',
      fuelUsage: 'Diagram Pemakaian BBM',
      mapTitle: 'Peta Kapal',
      detailUsage: 'Detail Pemakaian BBM',
      shipUsageTable: 'Detail Penggunaan Kapal',
      fuelPrediction: 'Prediksi Konsumsi Bahan Bakar',
      arrivalPrediction: 'Estimasi Waktu Kedatangan',
      predicted: 'Prediksi',
      confidence: 'Kepercayaan',
      remaining: 'Jarak Tersisa',
      daily: 'Harian',
      weekly: 'Mingguan',
      monthly: 'Bulanan',
      search: 'Cari',
      showing: 'Menampilkan',
      of: 'dari',
      entries: 'entri',
      shipName: 'Nama Kapal',
      tank: 'Tangki',
      volume: 'Volume',
      usage: 'Pemakaian',
      datetime: 'Tanggal & Waktu',
      status: 'Status',
      liters: 'liter',
      hours: 'jam',
      nm: 'NM',
      knots: 'knot',
      vessel: 'Kapal',
    }
  };

  const t = translations[language];

  useEffect(() => {
    const mockShips = [
      { id: '1', name: 'Rasuna Baruna', lat: -3.789, lon: 114.523, status: 'in_progress', route: 'Jepara-Taboneo', color: '#3b82f6' },
      { id: '2', name: 'Latifah Baruna', lat: -5.234, lon: 112.456, status: 'in_progress', route: 'Taboneo-Labuan Bajo', color: '#10b981' },
      { id: '3', name: 'Martha Baruna', lat: -6.234, lon: 110.789, status: 'in_port', route: 'Jepara Port', color: '#f59e0b' },
      { id: '4', name: 'Meutia Baruna', lat: -3.234, lon: 116.123, status: 'in_progress', route: 'Kalimantan-Sulawesi', color: '#ef4444' },
    ];
    setShips(mockShips);
    setSelectedShip(mockShips[0]);

    const initialLayers = {};
    mockShips.forEach(ship => {
      initialLayers[ship.id] = true;
    });
    setActiveShipLayer(initialLayers);

    setFuelPrediction({
      predicted: 12500,
      lower: 11250,
      upper: 13750,
      confidence: 0.85,
      distance: 450,
      eta: 36
    });

    setArrivalPrediction({
      predicted: new Date(Date.now() + 36 * 3600 * 1000),
      lower: 2,
      upper: 2,
      confidence: 0.82,
      distance: 450,
      speed: 12.5
    });

    const mockVoyages = [
      { id: 1, shipName: 'Rasuna Baruna', tank: 'TK-1', volume: 10000, usage: 5500, datetime: '14 Nov 2021, 07:08:24', status: 'Active' },
      { id: 2, shipName: 'Rasuna Baruna', tank: 'TK-2', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 02:12', status: 'Active' },
      { id: 3, shipName: 'Latifah Baruna', tank: 'TK-5', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 03:12', status: 'Active' },
      { id: 4, shipName: 'Latifah Baruna', tank: 'TK-3', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 04:14', status: 'Active' },
      { id: 5, shipName: 'Latifah Baruna', tank: 'TK-2', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 05:23', status: 'Active' },
      { id: 6, shipName: 'Martha Baruna', tank: 'TK-1', volume: 10000, usage: 4200, datetime: '13 Nov 2021, 06:15', status: 'Active' },
      { id: 7, shipName: 'Meutia Baruna', tank: 'TK-4', volume: 10000, usage: 6800, datetime: '13 Nov 2021, 08:45', status: 'Active' },
    ];
    setVoyageData(mockVoyages);
  }, []);

  const bbmUsageData = {
    daily: [
      { day: 'Mon', usage: 500 },
      { day: 'Tue', usage: 250 },
      { day: 'Wed', usage: 700 },
      { day: 'Thu', usage: 450 },
      { day: 'Fri', usage: 600 },
      { day: 'Sat', usage: 800 },
      { day: 'Sun', usage: 550 },
    ],
    weekly: [
      { day: 'Week 1', usage: 3200 },
      { day: 'Week 2', usage: 2800 },
      { day: 'Week 3', usage: 3500 },
      { day: 'Week 4', usage: 3000 },
    ],
    monthly: [
      { day: 'Jan', usage: 12000 },
      { day: 'Feb', usage: 11500 },
      { day: 'Mar', usage: 13000 },
      { day: 'Apr', usage: 12500 },
    ]
  };

  const tankUsage = [
    { name: 'Tank 1', percentage: 81 },
    { name: 'Tank 2', percentage: 81 },
    { name: 'Tank 3', percentage: 87 },
    { name: 'Tank 4', percentage: 87 },
    { name: 'Tank 5', percentage: 87 },
  ];

  const filteredData = voyageData.filter(item =>
    item.shipName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.tank.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const paginatedData = filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');
  const toggleLanguage = () => setLanguage(language === 'en' ? 'id' : 'en');

  const bgColor = theme === 'dark' ? 'bg-gray-900' : 'bg-gray-50';
  const cardBg = theme === 'dark' ? 'bg-gray-800' : 'bg-white';
  const textColor = theme === 'dark' ? 'text-gray-100' : 'text-gray-900';
  const textSecondary = theme === 'dark' ? 'text-gray-400' : 'text-gray-600';
  const borderColor = theme === 'dark' ? 'border-gray-700' : 'border-gray-200';

  return (
    <div className={`min-h-screen ${bgColor} ${textColor}`}>
      <div className={`${cardBg} border-b ${borderColor} px-6 py-4`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Ship className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-2xl font-bold">TytoAlba</h1>
              <p className={`text-sm ${textSecondary}`}>{t.fuelManagement}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={toggleLanguage}
              className={`p-2 rounded-lg ${cardBg} border ${borderColor} hover:bg-opacity-80 transition`}
            >
              <Globe className="w-5 h-5" />
            </button>
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg ${cardBg} border ${borderColor} hover:bg-opacity-80 transition`}
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <div className={`px-4 py-2 rounded-lg ${cardBg} border ${borderColor}`}>
              <span className={`text-sm ${textSecondary}`}>2025-10-09 10:00 WIB</span>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <div className="flex items-center gap-3 mb-4">
              <Fuel className="w-6 h-6 text-blue-500" />
              <h2 className="text-xl font-semibold">{t.fuelPrediction}</h2>
            </div>
            {fuelPrediction && (
              <div className="space-y-4">
                <div className="flex items-end gap-2">
                  <span className="text-4xl font-bold text-blue-500">{fuelPrediction.predicted.toLocaleString()}</span>
                  <span className={`text-lg ${textSecondary} mb-1`}>{t.liters}</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className={`text-sm ${textSecondary}`}>{t.confidence} Score</p>
                    <p className="text-lg font-semibold">{(fuelPrediction.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className={`text-sm ${textSecondary}`}>{t.remaining}</p>
                    <p className="text-lg font-semibold">{fuelPrediction.distance} {t.nm}</p>
                  </div>
                </div>
                <div className={`p-3 rounded ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <p className={`text-xs ${textSecondary} mb-1`}>Range (±10%)</p>
                  <p className="text-sm">{fuelPrediction.lower.toLocaleString()} - {fuelPrediction.upper.toLocaleString()} {t.liters}</p>
                </div>
              </div>
            )}
          </div>

          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <div className="flex items-center gap-3 mb-4">
              <Clock className="w-6 h-6 text-green-500" />
              <h2 className="text-xl font-semibold">{t.arrivalPrediction}</h2>
            </div>
            {arrivalPrediction && (
              <div className="space-y-4">
                <div>
                  <p className={`text-sm ${textSecondary} mb-1`}>{t.predicted} ETA</p>
                  <p className="text-2xl font-bold text-green-500">
                    {arrivalPrediction.predicted.toLocaleDateString()} {arrivalPrediction.predicted.toLocaleTimeString()}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className={`text-sm ${textSecondary}`}>{t.confidence}</p>
                    <p className="text-lg font-semibold">{(arrivalPrediction.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className={`text-sm ${textSecondary}`}>Speed</p>
                    <p className="text-lg font-semibold">{arrivalPrediction.speed} {t.knots}</p>
                  </div>
                </div>
                <div className={`p-3 rounded ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <p className={`text-xs ${textSecondary} mb-1`}>Time Window</p>
                  <p className="text-sm">± {arrivalPrediction.lower} {t.hours}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <h3 className="text-lg font-semibold mb-4">{t.fuelUsage}</h3>
            <div className="space-y-4">
              {tankUsage.map((tank, idx) => (
                <div key={idx}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">{tank.name}</span>
                    <span className="text-sm font-semibold">{tank.percentage}%</span>
                  </div>
                  <div className={`h-2 rounded-full ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}>
                    <div
                      className="h-2 rounded-full bg-blue-500"
                      style={{ width: `${tank.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <h3 className="text-lg font-semibold mb-4">{t.mapTitle}</h3>
            <div className="relative w-full h-64 bg-blue-100 rounded overflow-hidden">
              <svg className="w-full h-full" viewBox="0 0 400 300">
                <path d="M50,150 Q100,120 150,140 T250,130 T350,150" stroke="#94a3b8" strokeWidth="2" fill="none" strokeDasharray="5,5" />
                {ships.filter(ship => activeShipLayer[ship.id]).map((ship, idx) => {
                  const x = 50 + idx * 100;
                  const y = 120 + Math.sin(idx) * 30;
                  return (
                    <g key={ship.id}>
                      <circle cx={x} cy={y} r="6" fill={ship.color} />
                      <text x={x} y={y - 12} fontSize="10" fill={theme === 'dark' ? '#fff' : '#000'} textAnchor="middle">
                        {ship.name.split(' ')[0]}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            
            <div className="mt-4 space-y-2">
              <p className={`text-xs font-semibold ${textSecondary}`}>{t.vessel}</p>
              {ships.map(ship => (
                <label key={ship.id} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={activeShipLayer[ship.id]}
                    onChange={(e) => setActiveShipLayer({...activeShipLayer, [ship.id]: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ship.color }} />
                  <span className="text-sm">{ship.name}</span>
                </label>
              ))}
            </div>
          </div>

          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">{t.detailUsage}</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setTimeRange('daily')}
                  className={`px-3 py-1 text-xs rounded ${timeRange === 'daily' ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {t.daily}
                </button>
                <button
                  onClick={() => setTimeRange('weekly')}
                  className={`px-3 py-1 text-xs rounded ${timeRange === 'weekly' ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {t.weekly}
                </button>
                <button
                  onClick={() => setTimeRange('monthly')}
                  className={`px-3 py-1 text-xs rounded ${timeRange === 'monthly' ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {t.monthly}
                </button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bbmUsageData[timeRange]}>
                <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                <XAxis dataKey="day" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme === 'dark' ? '#1f2937' : '#fff',
                    border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                    color: theme === 'dark' ? '#fff' : '#000'
                  }}
                />
                <Bar dataKey="usage" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={`${cardBg} rounded-lg border ${borderColor} p-6 mt-6`}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">{t.shipUsageTable}</h3>
            <input
              type="text"
              placeholder={t.search}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`px-4 py-2 rounded border ${borderColor} ${theme === 'dark' ? 'bg-gray-700' : 'bg-white'}`}
            />
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className={`${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold">No</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.shipName}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.tank}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.volume}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.usage}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.datetime}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.status}</th>
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((item, idx) => (
                  <tr key={item.id} className={`border-b ${borderColor} hover:bg-opacity-50 ${theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}`}>
                    <td className="px-4 py-3 text-sm">{(currentPage - 1) * itemsPerPage + idx + 1}</td>
                    <td className="px-4 py-3 text-sm">{item.shipName}</td>
                    <td className="px-4 py-3 text-sm">{item.tank}</td>
                    <td className="px-4 py-3 text-sm">{item.volume.toLocaleString()}</td>
                    <td className="px-4 py-3 text-sm">{item.usage.toLocaleString()}</td>
                    <td className="px-4 py-3 text-sm">{item.datetime}</td>
                    <td className="px-4 py-3 text-sm">
                      <span className="px-2 py-1 text-xs rounded bg-green-500 bg-opacity-20 text-green-500">
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex justify-between items-center mt-4">
            <p className={`text-sm ${textSecondary}`}>
              {t.showing} {((currentPage - 1) * itemsPerPage) + 1} - {Math.min(currentPage * itemsPerPage, filteredData.length)} {t.of} {filteredData.length} {t.entries}
            </p>
            <div className="flex gap-2">
              {Array.from({ length: totalPages }, (_, i) => (
                <button
                  key={i + 1}
                  onClick={() => setCurrentPage(i + 1)}
                  className={`px-3 py-1 rounded ${currentPage === i + 1 ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TytoAlbaDashboard;aimport React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Ship, Fuel, Navigation, Clock, TrendingUp, Sun, Moon, Globe, MapPin } from 'lucide-react';

const TytoAlbaDashboard = () => {
  const [theme, setTheme] = useState('dark');
  const [language, setLanguage] = useState('en');
  const [selectedShip, setSelectedShip] = useState(null);
  const [timeRange, setTimeRange] = useState('weekly');
  const [ships, setShips] = useState([]);
  const [voyageData, setVoyageData] = useState([]);
  const [fuelPrediction, setFuelPrediction] = useState(null);
  const [arrivalPrediction, setArrivalPrediction] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [activeShipLayer, setActiveShipLayer] = useState({});
  const itemsPerPage = 5;

  const translations = {
    en: {
      dashboard: 'Dashboard',
      fuelManagement: 'Fuel Management System',
      ships: 'Ships',
      fuelUsage: 'BBM Usage Diagram',
      mapTitle: 'Ship Map',
      detailUsage: 'Detail BBM Usage',
      shipUsageTable: 'Ship Usage Detail',
      fuelPrediction: 'Fuel Consumption Prediction',
      arrivalPrediction: 'Estimated Time of Arrival',
      predicted: 'Predicted',
      confidence: 'Confidence',
      remaining: 'Remaining Distance',
      daily: 'Daily',
      weekly: 'Weekly',
      monthly: 'Monthly',
      search: 'Search',
      showing: 'Showing',
      of: 'of',
      entries: 'entries',
      shipName: 'Ship Name',
      tank: 'Tank',
      volume: 'Volume',
      usage: 'Usage',
      datetime: 'Date & Time',
      status: 'Status',
      liters: 'liters',
      hours: 'hours',
      nm: 'NM',
      knots: 'knots',
      vessel: 'Vessel',
    },
    id: {
      dashboard: 'Dasbor',
      fuelManagement: 'Sistem Manajemen Bahan Bakar',
      ships: 'Kapal',
      fuelUsage: 'Diagram Pemakaian BBM',
      mapTitle: 'Peta Kapal',
      detailUsage: 'Detail Pemakaian BBM',
      shipUsageTable: 'Detail Penggunaan Kapal',
      fuelPrediction: 'Prediksi Konsumsi Bahan Bakar',
      arrivalPrediction: 'Estimasi Waktu Kedatangan',
      predicted: 'Prediksi',
      confidence: 'Kepercayaan',
      remaining: 'Jarak Tersisa',
      daily: 'Harian',
      weekly: 'Mingguan',
      monthly: 'Bulanan',
      search: 'Cari',
      showing: 'Menampilkan',
      of: 'dari',
      entries: 'entri',
      shipName: 'Nama Kapal',
      tank: 'Tangki',
      volume: 'Volume',
      usage: 'Pemakaian',
      datetime: 'Tanggal & Waktu',
      status: 'Status',
      liters: 'liter',
      hours: 'jam',
      nm: 'NM',
      knots: 'knot',
      vessel: 'Kapal',
    }
  };

  const t = translations[language];

  useEffect(() => {
    const mockShips = [
      { id: '1', name: 'Rasuna Baruna', lat: -3.789, lon: 114.523, status: 'in_progress', route: 'Jepara-Taboneo', color: '#3b82f6' },
      { id: '2', name: 'Latifah Baruna', lat: -5.234, lon: 112.456, status: 'in_progress', route: 'Taboneo-Labuan Bajo', color: '#10b981' },
      { id: '3', name: 'Martha Baruna', lat: -6.234, lon: 110.789, status: 'in_port', route: 'Jepara Port', color: '#f59e0b' },
      { id: '4', name: 'Meutia Baruna', lat: -3.234, lon: 116.123, status: 'in_progress', route: 'Kalimantan-Sulawesi', color: '#ef4444' },
    ];
    setShips(mockShips);
    setSelectedShip(mockShips[0]);

    const initialLayers = {};
    mockShips.forEach(ship => {
      initialLayers[ship.id] = true;
    });
    setActiveShipLayer(initialLayers);

    setFuelPrediction({
      predicted: 12500,
      lower: 11250,
      upper: 13750,
      confidence: 0.85,
      distance: 450,
      eta: 36
    });

    setArrivalPrediction({
      predicted: new Date(Date.now() + 36 * 3600 * 1000),
      lower: 2,
      upper: 2,
      confidence: 0.82,
      distance: 450,
      speed: 12.5
    });

    const mockVoyages = [
      { id: 1, shipName: 'Rasuna Baruna', tank: 'TK-1', volume: 10000, usage: 5500, datetime: '14 Nov 2021, 07:08:24', status: 'Active' },
      { id: 2, shipName: 'Rasuna Baruna', tank: 'TK-2', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 02:12', status: 'Active' },
      { id: 3, shipName: 'Latifah Baruna', tank: 'TK-5', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 03:12', status: 'Active' },
      { id: 4, shipName: 'Latifah Baruna', tank: 'TK-3', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 04:14', status: 'Active' },
      { id: 5, shipName: 'Latifah Baruna', tank: 'TK-2', volume: 10000, usage: 5900, datetime: '14 Nov 2021, 05:23', status: 'Active' },
      { id: 6, shipName: 'Martha Baruna', tank: 'TK-1', volume: 10000, usage: 4200, datetime: '13 Nov 2021, 06:15', status: 'Active' },
      { id: 7, shipName: 'Meutia Baruna', tank: 'TK-4', volume: 10000, usage: 6800, datetime: '13 Nov 2021, 08:45', status: 'Active' },
    ];
    setVoyageData(mockVoyages);
  }, []);

  const bbmUsageData = {
    daily: [
      { day: 'Mon', usage: 500 },
      { day: 'Tue', usage: 250 },
      { day: 'Wed', usage: 700 },
      { day: 'Thu', usage: 450 },
      { day: 'Fri', usage: 600 },
      { day: 'Sat', usage: 800 },
      { day: 'Sun', usage: 550 },
    ],
    weekly: [
      { day: 'Week 1', usage: 3200 },
      { day: 'Week 2', usage: 2800 },
      { day: 'Week 3', usage: 3500 },
      { day: 'Week 4', usage: 3000 },
    ],
    monthly: [
      { day: 'Jan', usage: 12000 },
      { day: 'Feb', usage: 11500 },
      { day: 'Mar', usage: 13000 },
      { day: 'Apr', usage: 12500 },
    ]
  };

  const tankUsage = [
    { name: 'Tank 1', percentage: 81 },
    { name: 'Tank 2', percentage: 81 },
    { name: 'Tank 3', percentage: 87 },
    { name: 'Tank 4', percentage: 87 },
    { name: 'Tank 5', percentage: 87 },
  ];

  const filteredData = voyageData.filter(item =>
    item.shipName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.tank.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const paginatedData = filteredData.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const toggleTheme = () => setTheme(theme === 'dark' ? 'light' : 'dark');
  const toggleLanguage = () => setLanguage(language === 'en' ? 'id' : 'en');

  const bgColor = theme === 'dark' ? 'bg-gray-900' : 'bg-gray-50';
  const cardBg = theme === 'dark' ? 'bg-gray-800' : 'bg-white';
  const textColor = theme === 'dark' ? 'text-gray-100' : 'text-gray-900';
  const textSecondary = theme === 'dark' ? 'text-gray-400' : 'text-gray-600';
  const borderColor = theme === 'dark' ? 'border-gray-700' : 'border-gray-200';

  return (
    <div className={`min-h-screen ${bgColor} ${textColor}`}>
      <div className={`${cardBg} border-b ${borderColor} px-6 py-4`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Ship className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-2xl font-bold">TytoAlba</h1>
              <p className={`text-sm ${textSecondary}`}>{t.fuelManagement}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={toggleLanguage}
              className={`p-2 rounded-lg ${cardBg} border ${borderColor} hover:bg-opacity-80 transition`}
            >
              <Globe className="w-5 h-5" />
            </button>
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg ${cardBg} border ${borderColor} hover:bg-opacity-80 transition`}
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <div className={`px-4 py-2 rounded-lg ${cardBg} border ${borderColor}`}>
              <span className={`text-sm ${textSecondary}`}>2025-10-09 10:00 WIB</span>
            </div>
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <div className="flex items-center gap-3 mb-4">
              <Fuel className="w-6 h-6 text-blue-500" />
              <h2 className="text-xl font-semibold">{t.fuelPrediction}</h2>
            </div>
            {fuelPrediction && (
              <div className="space-y-4">
                <div className="flex items-end gap-2">
                  <span className="text-4xl font-bold text-blue-500">{fuelPrediction.predicted.toLocaleString()}</span>
                  <span className={`text-lg ${textSecondary} mb-1`}>{t.liters}</span>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className={`text-sm ${textSecondary}`}>{t.confidence} Score</p>
                    <p className="text-lg font-semibold">{(fuelPrediction.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className={`text-sm ${textSecondary}`}>{t.remaining}</p>
                    <p className="text-lg font-semibold">{fuelPrediction.distance} {t.nm}</p>
                  </div>
                </div>
                <div className={`p-3 rounded ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <p className={`text-xs ${textSecondary} mb-1`}>Range (±10%)</p>
                  <p className="text-sm">{fuelPrediction.lower.toLocaleString()} - {fuelPrediction.upper.toLocaleString()} {t.liters}</p>
                </div>
              </div>
            )}
          </div>

          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <div className="flex items-center gap-3 mb-4">
              <Clock className="w-6 h-6 text-green-500" />
              <h2 className="text-xl font-semibold">{t.arrivalPrediction}</h2>
            </div>
            {arrivalPrediction && (
              <div className="space-y-4">
                <div>
                  <p className={`text-sm ${textSecondary} mb-1`}>{t.predicted} ETA</p>
                  <p className="text-2xl font-bold text-green-500">
                    {arrivalPrediction.predicted.toLocaleDateString()} {arrivalPrediction.predicted.toLocaleTimeString()}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className={`text-sm ${textSecondary}`}>{t.confidence}</p>
                    <p className="text-lg font-semibold">{(arrivalPrediction.confidence * 100).toFixed(0)}%</p>
                  </div>
                  <div>
                    <p className={`text-sm ${textSecondary}`}>Speed</p>
                    <p className="text-lg font-semibold">{arrivalPrediction.speed} {t.knots}</p>
                  </div>
                </div>
                <div className={`p-3 rounded ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
                  <p className={`text-xs ${textSecondary} mb-1`}>Time Window</p>
                  <p className="text-sm">± {arrivalPrediction.lower} {t.hours}</p>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <h3 className="text-lg font-semibold mb-4">{t.fuelUsage}</h3>
            <div className="space-y-4">
              {tankUsage.map((tank, idx) => (
                <div key={idx}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm">{tank.name}</span>
                    <span className="text-sm font-semibold">{tank.percentage}%</span>
                  </div>
                  <div className={`h-2 rounded-full ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}>
                    <div
                      className="h-2 rounded-full bg-blue-500"
                      style={{ width: `${tank.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <h3 className="text-lg font-semibold mb-4">{t.mapTitle}</h3>
            <div className="relative w-full h-64 bg-blue-100 rounded overflow-hidden">
              <svg className="w-full h-full" viewBox="0 0 400 300">
                <path d="M50,150 Q100,120 150,140 T250,130 T350,150" stroke="#94a3b8" strokeWidth="2" fill="none" strokeDasharray="5,5" />
                {ships.filter(ship => activeShipLayer[ship.id]).map((ship, idx) => {
                  const x = 50 + idx * 100;
                  const y = 120 + Math.sin(idx) * 30;
                  return (
                    <g key={ship.id}>
                      <circle cx={x} cy={y} r="6" fill={ship.color} />
                      <text x={x} y={y - 12} fontSize="10" fill={theme === 'dark' ? '#fff' : '#000'} textAnchor="middle">
                        {ship.name.split(' ')[0]}
                      </text>
                    </g>
                  );
                })}
              </svg>
            </div>
            
            <div className="mt-4 space-y-2">
              <p className={`text-xs font-semibold ${textSecondary}`}>{t.vessel}</p>
              {ships.map(ship => (
                <label key={ship.id} className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={activeShipLayer[ship.id]}
                    onChange={(e) => setActiveShipLayer({...activeShipLayer, [ship.id]: e.target.checked})}
                    className="w-4 h-4"
                  />
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: ship.color }} />
                  <span className="text-sm">{ship.name}</span>
                </label>
              ))}
            </div>
          </div>

          <div className={`${cardBg} rounded-lg border ${borderColor} p-6`}>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">{t.detailUsage}</h3>
              <div className="flex gap-2">
                <button
                  onClick={() => setTimeRange('daily')}
                  className={`px-3 py-1 text-xs rounded ${timeRange === 'daily' ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {t.daily}
                </button>
                <button
                  onClick={() => setTimeRange('weekly')}
                  className={`px-3 py-1 text-xs rounded ${timeRange === 'weekly' ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {t.weekly}
                </button>
                <button
                  onClick={() => setTimeRange('monthly')}
                  className={`px-3 py-1 text-xs rounded ${timeRange === 'monthly' ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {t.monthly}
                </button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={bbmUsageData[timeRange]}>
                <CartesianGrid strokeDasharray="3 3" stroke={theme === 'dark' ? '#374151' : '#e5e7eb'} />
                <XAxis dataKey="day" stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                <YAxis stroke={theme === 'dark' ? '#9ca3af' : '#6b7280'} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: theme === 'dark' ? '#1f2937' : '#fff',
                    border: `1px solid ${theme === 'dark' ? '#374151' : '#e5e7eb'}`,
                    color: theme === 'dark' ? '#fff' : '#000'
                  }}
                />
                <Bar dataKey="usage" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className={`${cardBg} rounded-lg border ${borderColor} p-6 mt-6`}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold">{t.shipUsageTable}</h3>
            <input
              type="text"
              placeholder={t.search}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className={`px-4 py-2 rounded border ${borderColor} ${theme === 'dark' ? 'bg-gray-700' : 'bg-white'}`}
            />
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className={`${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-100'}`}>
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold">No</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.shipName}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.tank}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.volume}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.usage}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.datetime}</th>
                  <th className="px-4 py-3 text-left text-sm font-semibold">{t.status}</th>
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((item, idx) => (
                  <tr key={item.id} className={`border-b ${borderColor} hover:bg-opacity-50 ${theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-50'}`}>
                    <td className="px-4 py-3 text-sm">{(currentPage - 1) * itemsPerPage + idx + 1}</td>
                    <td className="px-4 py-3 text-sm">{item.shipName}</td>
                    <td className="px-4 py-3 text-sm">{item.tank}</td>
                    <td className="px-4 py-3 text-sm">{item.volume.toLocaleString()}</td>
                    <td className="px-4 py-3 text-sm">{item.usage.toLocaleString()}</td>
                    <td className="px-4 py-3 text-sm">{item.datetime}</td>
                    <td className="px-4 py-3 text-sm">
                      <span className="px-2 py-1 text-xs rounded bg-green-500 bg-opacity-20 text-green-500">
                        {item.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="flex justify-between items-center mt-4">
            <p className={`text-sm ${textSecondary}`}>
              {t.showing} {((currentPage - 1) * itemsPerPage) + 1} - {Math.min(currentPage * itemsPerPage, filteredData.length)} {t.of} {filteredData.length} {t.entries}
            </p>
            <div className="flex gap-2">
              {Array.from({ length: totalPages }, (_, i) => (
                <button
                  key={i + 1}
                  onClick={() => setCurrentPage(i + 1)}
                  className={`px-3 py-1 rounded ${currentPage === i + 1 ? 'bg-blue-500 text-white' : `${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}`}`}
                >
                  {i + 1}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TytoAlbaDashboard;
