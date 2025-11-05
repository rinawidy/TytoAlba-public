# TytoAlba Wireframes

This directory contains all wireframe designs for the TytoAlba Predictive Monitoring System, created using PlantUML's Salt (wireframe) syntax.

## Wireframe Files

### Desktop Web Application

1. **`01-dashboard-fleet-overview.puml`** - Main dashboard with live map and fleet status
2. **`02-ship-detail-view.puml`** - Individual ship details with route, ETA, and fuel info
3. **`03-eta-prediction-page.puml`** - LSTM-based ETA prediction interface
4. **`04-fuel-monitoring-page.puml`** - Real-time fuel monitoring and consumption tracking
5. **`05-reports-analytics.puml`** - Reports generation and analytics dashboard
6. **`06-admin-panel.puml`** - System administration and configuration

### Mobile Web (Responsive Design)

7. **`07-mobile-responsive-web.puml`** - Mobile web view (9 responsive screens: Dashboard, Ship Detail, ETA Prediction, Fuel Monitor, Reports, Menu Sidebar, Alerts, Manual Fuel Entry, Login)

## How to Render Wireframes

Wireframes use the same rendering methods as UML diagrams.

### Method 1: Online PlantUML Editor (Quickest)

1. Go to [PlantUML Online Editor](http://www.plantuml.com/plantuml/uml/)
2. Copy content from any `.puml` file
3. Paste and view instantly
4. Download as PNG/SVG

### Method 2: VS Code Extension (Best for Development)

1. Install PlantUML extension:
   ```bash
   code --install-extension jebbs.plantuml
   ```

2. Install Graphviz:
   - **Windows:** `choco install graphviz`
   - **macOS:** `brew install graphviz`
   - **Linux:** `sudo apt-get install graphviz`

3. Open any `.puml` file
4. Press `Alt+D` to preview
5. Right-click → "Export Current Diagram" to save

### Method 3: Command Line

```bash
# Download PlantUML JAR
wget https://github.com/plantuml/plantuml/releases/download/v1.2023.13/plantuml-1.2023.13.jar

# Render all wireframes as PNG
java -jar plantuml.jar *.puml

# Render as SVG (recommended for web)
java -jar plantuml.jar -tsvg *.puml
```

### Method 4: Docker

```bash
cd docs/wireframes

# Render as PNG
docker run --rm -v $(pwd):/data plantuml/plantuml:latest -tpng /data/*.puml

# Render as SVG
docker run --rm -v $(pwd):/data plantuml/plantuml:latest -tsvg /data/*.puml
```

## Wireframe Descriptions

### 1. Dashboard / Fleet Overview
**File:** `01-dashboard-fleet-overview.puml`

**Purpose:** Main landing page for fleet managers and administrators.

**Key Components:**
- **Top Navigation:** Logo, user profile, notifications (3), logout
- **Sidebar Menu:** Dashboard, Ships, Reports, Analytics, Settings, Admin Panel
- **Live Map:** Interactive Leaflet.js map showing Indonesia with:
  - Ship markers (🚢) with different colors
  - Solid green lines (traversed paths)
  - Dotted gray lines (remaining routes)
  - Port markers (Jepara, Taboneo, Labuan Bajo, Makassar)
  - Map controls (Zoom In/Out, Reset View, Full Screen)
- **Quick Stats Bar:** Total Ships (12), In Transit (9), In Port (3), Delayed (1)
- **Active Alerts:** Low fuel warnings, route updates with actions
- **Fleet Status Table:**
  - Columns: No, Ship Name, Status, Location, Destination, ETA, Fuel, Actions
  - 5 ships displayed with status indicators (●)
  - Color-coded status: Blue (In Transit), Green (In Port), Red (Delayed)
- **Footer:** Last updated timestamp, Refresh Data button

**User Roles:** Fleet Manager, Administrator

---

### 2. Ship Detail View
**File:** `02-ship-detail-view.puml`

**Purpose:** Comprehensive view of a single ship's current status and history.

**Key Components:**
- **Header:** Ship name (Rasuna Baruna), status badge, MMSI number, Back to Fleet button
- **Breadcrumb Navigation:** Dashboard → Ships → Rasuna Baruna
- **Ship Information Panel:**
  - Vessel specs: Type, Tonnage, Length, Draft, Max Speed, Fuel Capacity
  - Current metrics: Speed, Fuel Level, Last Update
- **Current Position Card:**
  - GPS coordinates (Lat/Lon)
  - Course heading (045° NE)
  - "View on Map" button
- **Live Route Map:**
  - Origin → Current Position → Waypoints → Destination
  - Visual representation of traversed (solid green) vs remaining (dotted gray) route
  - Distance remaining, progress percentage
  - Weather/Traffic overlay toggles
- **ETA Prediction (LSTM):**
  - Predicted arrival date/time
  - Time remaining
  - Confidence score with progress bar (92%)
  - 95% confidence interval (± 2h 24m)
  - Predicted fuel on arrival
  - Refresh and History buttons
- **Current Fuel Status:**
  - ASCII fuel gauge visualization (68%)
  - Current level (12,500L)
  - Consumption rate (167 L/h)
  - Last reading timestamp and source (Sensor/Manual)
  - Manual Entry button
- **Fuel Consumption Chart:**
  - Line chart showing 7-day fuel level trend
  - Time period selectors (24h, 7d, 30d, Custom)
- **Weather Conditions Table:**
  - Current position + upcoming waypoints
  - Condition icons, temperature, wind, wave height
- **Recent Activity Log:**
  - Chronological event list (ETA updates, fuel readings, position updates)
- **Action Buttons:** Export Report (PDF), Email Report, Configure Alerts

**User Roles:** Captain, Fleet Manager, Administrator

---

### 3. ETA Prediction Page
**File:** `03-eta-prediction-page.puml`

**Purpose:** Dedicated interface for generating and analyzing ETA predictions using LSTM model.

**Key Components:**
- **Ship Selection Panel:**
  - Dropdown list of all ships (filtered by status: In Transit, In Port, Delayed)
  - Selected ship summary: MMSI, position, destination, distance, speed
  - "Generate Prediction" button
- **Prediction Results Section:**
  - Success message with checkmark
  - **Large ETA Display:** Date and time (Oct 22, 2024 at 14:30 WIB)
  - Time remaining (24h 30m)
  - **Confidence Metrics:**
    - Model confidence percentage (92%) with progress bar
    - 95% confidence interval (earliest/latest times)
    - Model version (lstm-v2.3.1)
    - Last training date
    - Training accuracy metrics (MAE, MAPE)
  - **Predicted Fuel on Arrival:**
    - Fuel gauge (42%, 8,500L)
    - Current vs predicted comparison
    - Consumption rate
    - Fuel sufficiency status (green checkmark)
    - Recommended action
- **Prediction Factors Analysis Table:**
  - 8 factors with current values, impact assessment, and weight bars
  - Distance, Speed, Weather, Sea Current, Wave Height, Historical Performance, Vessel Characteristics, Traffic Density
- **Route Visualization with Prediction:**
  - Detailed route diagram with timestamps
  - Weather icons and conditions at each waypoint
  - Progress percentage and time elapsed/remaining
  - Alternative routes button
- **Prediction History Table:**
  - Past predictions with actual outcomes
  - Accuracy comparison (green ✓ for accurate, orange △ for delayed)
  - Current prediction highlighted
- **Model Performance Insights:**
  - 30-day accuracy statistics
  - MAE, MAPE, R² score
  - Predictions within 1h/2h percentages
  - Performance status (above/below target)
- **Alerts & Recommendations:**
  - Weather advisories
  - Route status
  - Fuel advisories
- **Action Buttons:** Detailed Analysis, Export Report (PDF), Email to Stakeholders, Set ETA Alert

**User Roles:** Fleet Manager, Captain (view only)

---

### 4. Fuel Monitoring Page
**File:** `04-fuel-monitoring-page.puml`

**Purpose:** Real-time fuel monitoring with historical trends and manual entry capability.

**Key Components:**
- **Ship Selector:**
  - Dropdown with all ships
  - "All Ships" option for fleet-wide view
- **Quick Metrics Panel:**
  - Current fuel level (12,500L, 68%)
  - Last reading timestamp and source (Auto Sensor/Manual)
  - Consumption rate (167 L/h)
  - Tank health status (● Normal/Degraded/Critical)
  - Sensor status (● Online/Offline)
  - Manual Entry and Refresh buttons
- **Live Fuel Gauge:**
  - ASCII tank visualization with fill percentage
  - Current volume (12,500L) and capacity (50,000L)
  - Threshold indicators:
    - 90% (Critical High)
    - 20% (Low Fuel Warning - orange)
    - 10% (Critical Low - red)
  - Status message (green ✓ Fuel level normal)
- **Consumption Statistics:**
  - Last 24h: Total consumed, average rate, peak/lowest rates
  - Last 7 days: Total consumed, daily average, efficiency (L/nm)
  - Predictions: Depletion time, arrival fuel estimate, refuel required status
- **Fuel Consumption Trend Chart (7 Days):**
  - Line graph showing fuel level over time
  - 50,000L to 0L scale
  - Legend: ● Sensor Reading, ○ Manual Entry, ━ Predicted Trend
  - Warning threshold line (orange)
  - Time period selectors, Export Chart button
- **Consumption Rate Analysis (24 Hours):**
  - Hourly consumption rate graph (0-200 L/h)
  - Average vs expected rate comparison
  - Variance calculation with warning if above threshold
- **Fuel Reading History Table:**
  - Timestamp, Fuel Level, Change, Rate, Source (Auto/Manual), Tank, Health
  - Color-coded health indicators
  - Manual entries highlighted
  - Load More and Export to Excel buttons
- **Sensor Health Monitoring Table:**
  - Sensor ID, Location (Tank 1/2/3), Status, Last Reading, Calibration Date
  - Action buttons: Calibrate, Test
- **Manual Fuel Entry Form:**
  - Tank selection dropdown
  - Fuel level input field
  - Reading method dropdown (Visual Gauge, Dipstick, Flow Meter)
  - Notes text area
  - Submit and Cancel buttons
  - Authentication note
- **Active Alerts Panel:**
  - Warning: High consumption (7.7% above expected)
  - Info: Sensor calibration due
  - Acknowledge/Investigate buttons
- **Fleet Fuel Comparison Table:**
  - All ships with current fuel, percentage, rate, efficiency
  - Status indicators and progress bars
- **Action Buttons:** Generate Fuel Report, Efficiency Analysis, Configure Thresholds, Alert Settings

**User Roles:** Captain (view + manual entry), Fleet Manager, Administrator

---

### 5. Reports & Analytics
**File:** `05-reports-analytics.puml`

**Purpose:** Generate comprehensive reports and view analytics across the fleet.

**Key Components:**
- **Report Filters:**
  - Report Type dropdown: Fuel Consumption, Voyage Performance, ETA Accuracy, Fleet Overview
  - Time Period dropdown: Last 7/30 Days, Last Quarter, Custom Range
  - Ships multi-select: All Ships or individual selection
  - Generate Report and Reset Filters buttons
- **Quick Stats (30 Days):**
  - Total Voyages (47)
  - Avg ETA Accuracy (96.2%)
  - Total Fuel Consumed (425,800L)
  - Avg Efficiency (2.1 L/nm)
  - Active Ships (12)
  - Delayed Voyages (3, 6.4%)
- **Fleet Performance Overview Chart:**
  - Horizontal bar chart of fuel consumption by ship (last 30 days)
  - Total and average per ship displayed
- **ETA Prediction Accuracy Analysis:**
  - Scatter plot of last 60 predictions
  - 80-100% accuracy range
  - ● Within 1h, ○ 1-2h variance
  - Avg MAE vs target comparison (28 min vs <30 min target ✓)
- **Fuel Efficiency Trends:**
  - Line chart over 6 weeks
  - L per nautical mile metric
  - Target line (2.2 L/nm)
  - Current status (2.1 L/nm ✓)
- **Detailed Voyage Report Table:**
  - Ship, Voyage ID, Route, Duration, Fuel Used, Efficiency, ETA Accuracy, Status
  - Color-coded status: ✓ Completed, △ Delayed
  - View All (47) and Export to Excel buttons
- **Performance Metrics Comparison Table:**
  - 6 key metrics comparing this month vs last month
  - Change percentage (green for improvement)
  - Target comparison
  - Status: ✓ Met or ✗ Not Met
- **ML Model Performance Dashboard:**
  - Current model version and details
  - Training date, samples, validation accuracy
  - Performance metrics: MAE, RMSE, MAPE, R² Score
  - Predictions this month count
  - Accuracy breakdown (within 1h/2h)
  - Status indicator (performing above target)
  - View Model Details and Retrain Model buttons
- **Top Performing Ships:**
  - Ranking by efficiency (L/nm)
  - Medal icons (🥇🥈🥉)
  - Voyage count
- **Least Performing Ships:**
  - Ships with highest consumption
  - Issues identified (e.g., "Engine maintenance needed")
- **Export & Sharing Options:**
  - Format selector: PDF Report, Excel Spreadsheet, CSV Data, JSON API
  - Report content checkboxes: Executive Summary, Charts & Graphs, Detailed Tables, Raw Data
  - Email To field
  - Schedule dropdown: One-time, Daily, Weekly, Monthly
  - Generate & Download, Email Report, Schedule Delivery buttons
- **Recent Generated Reports Table:**
  - Report Name, Type, Period, Generated timestamp
  - Actions: Download, Email, Delete

**User Roles:** Fleet Manager, Administrator

---

### 6. Admin Panel
**File:** `06-admin-panel.puml`

**Purpose:** System administration, configuration, and monitoring for administrators only.

**Key Components:**
- **System Health Overview:**
  - Overall Status indicator (● Healthy/Warning/Critical)
  - Component status list:
    - API Server (uptime %)
    - Database (connection status, response time)
    - MQTT Broker (messages/min)
    - ML Service (inference time)
    - Redis Cache (hit rate)
  - Last System Check timestamp
- **Quick Actions Grid:**
  - 8 action buttons: Manage Users, Add New Ship, Configure Sensors, Retrain ML Model, View System Logs, Database Backup, System Settings, Alert Configuration
- **User Management Table:**
  - Columns: Username, Email, Role, Ship Assignment, Last Login, Status, Actions
  - Roles: Administrator, Fleet Manager, Captain, Viewer
  - Status: ● Active/Inactive
  - Actions: Edit, Disable/Enable
  - Add New User, Import Users (CSV), Export User List buttons
- **Create/Edit User Form:**
  - Fields: Username, Email, Role dropdown, Assigned Ship dropdown
  - Permissions checkboxes (8 permissions)
  - Password fields with Generate Random button
  - Create User and Cancel buttons
- **Recent Admin Activities Log:**
  - Timestamp, User, Action
  - Chronological list of admin operations
- **ML Model Management:**
  - Current production model details (version, deployed date, performance)
  - Model versions table:
    - Version, Architecture, Training Date, Accuracy, Status, Actions
    - Statuses: ● Production, Archived, ● Testing
    - Actions: Details, Rollback, Restore, Deploy, Delete
  - Train New Model, Model Performance Dashboard, A/B Testing Setup buttons
- **Train New ML Model Form:**
  - Architecture dropdown: Bi-LSTM + Attention, Transformer, CNN-LSTM Hybrid
  - Training data options (all historical data or custom date range)
  - Hyperparameters: LSTM Units, Dropout, Learning Rate, Epochs
  - Validation split slider (20%)
  - Start Training, Load Previous Config, Cancel buttons
  - Estimated training time display
- **Training History & Logs Table:**
  - Version, Started, Duration, Status (✓ Success/✗ Failed)
- **Sensor & IoT Device Management Table:**
  - Sensor ID, Ship, Type, Status (● Online/Degraded/Offline), Last Data, Calibration Date, Firmware Version
  - Actions: Config, Test, Calibrate, Diagnose, Replace
  - Register New Sensor, Bulk Calibration Schedule, Firmware Update buttons
- **MQTT Broker Configuration:**
  - Status, Host, Ports (1883 plain, 8883 TLS)
  - Connected Clients count
  - Active Subscriptions count
  - Messages/min rate
  - Uptime
  - Topics list with subscription counts
  - View Live Messages, Restart Broker, Edit Config buttons
- **Database Configuration:**
  - Type (PostgreSQL 14.5 + TimescaleDB)
  - Host and port
  - Status
  - Database size
  - Tables count
  - Active connections (15/100)
  - Average query time
  - Largest tables with sizes and row counts
  - Last backup timestamp
  - Backup Now, Restore, Query Monitor buttons
- **System Monitoring & Logs:**
  - 6 metric charts (last 24h):
    - CPU Usage (%)
    - RAM Usage (GB)
    - Disk Usage (%)
    - API Requests/min
    - Error Rate (%)
    - Response Time (ms)
  - Line graphs with time axis (0h, 12h, 24h)
- **Recent System Logs Table:**
  - Timestamp, Level (INFO/WARN/ERROR), Service, Message
  - Color-coded log levels
  - View All Logs, Filter by Level, Export Logs, Configure Log Retention buttons
- **System Configuration Forms:**
  - **General Settings:** System Name, Timezone dropdown, Language dropdown
  - **Alert Thresholds:** Low Fuel Warning %, Critical Fuel %, Sensor Offline Alert minutes
  - **Data Retention Policies:** Retention days for Fuel Readings, GPS Positions, System Logs, Predictions
  - **Backup Settings:** Auto Backup checkbox, Frequency dropdown, Backup Time, Retention days
  - Save Changes and Reset to Defaults buttons

**User Roles:** Administrator only

---

### 7. Mobile Web (Responsive Design)
**File:** `07-mobile-responsive-web.puml`

**Purpose:** Responsive web application view optimized for mobile browsers (phone-sized screens).

**Key Difference from Desktop:** Same Vue.js application with responsive layouts for phone screens (320px-767px width), accessed via mobile browsers (Chrome, Safari, Firefox). NOT a native mobile app.

**Screens Included:**

#### Screen 1: Dashboard (Mobile Web)
- **Browser address bar visible:** `https://tytoalba...`
- Header: Hamburger menu (☰), TytoAlba logo, Notifications badge (🔔3)
- Quick stats row: 🚢12 ⛵9 ⚓3 ⚠1
- Map view (compressed, touch-enabled):
  - Ship markers (🚢 🚢 🚢)
  - Solid (═══>) and dotted (···>) route lines
  - Zoom controls adapted for touch
- Active Ships cards (vertically stacked):
  - Ship name, destination, ETA, fuel percentage
  - "View Details" button per ship
- "View All Ships" button
- Footer: Last update timestamp

#### Screen 2: Ship Detail (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: Back arrow (←), Ship name "Rasuna Baruna"
- Status badge: ● In Transit
- Ship info: MMSI, Speed
- Route map (vertical layout):
  - Origin (Jepara) → Current Position 🚢 → Destination (Taboneo)
  - Solid/dotted line visualization
- GPS coordinates display
- **ETA Prediction card:**
  - Date/time (Oct 22, 14:30)
  - Time remaining (24h 30m)
  - Confidence: 92% with progress bar
  - 95% CI: ±2h 24m
- **Fuel Status card:**
  - Horizontal progress bar (████████░░)
  - 12,500L, 68%
  - Rate: 167 L/h
  - Source: Sensor
- Action buttons: "Fuel History", "Weather Info"

#### Screen 3: ETA Prediction (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: Back arrow, "ETA Prediction"
- Ship selection dropdown (Rasuna Baruna▼)
- Current ship info (position, destination, distance)
- "🔮 Generate Prediction" button
- **Prediction Results card:**
  - ✓ Generated success message
  - Large ETA display (Oct 22, 14:30 WIB)
  - Time remaining (24h 30m)
  - Confidence score (92%) with bar
  - Confidence range (12:06 - 16:54, ±2h 24m)
- **Fuel on Arrival card:**
  - Fuel gauge (████░░░░░░)
  - 8,500L (42%)
  - ✓ Sufficient status
- Action buttons: "View Factors", "History"

#### Screen 4: Fuel Monitor (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: Back arrow, "Fuel Monitoring"
- Ship selector dropdown
- **Current Level card:**
  - Vertical fuel tank ASCII visualization
  - Fill percentage (68%)
  - Volume (12,500L)
- Last reading: 2 min ago, Source: Sensor
- **Consumption stats:**
  - Rate: 167 L/h
  - 24h: 4,008L
  - 7d: 28,560L
  - Efficiency: 2.1 L/nm
- Trend chart (7 days, simplified line graph)
- Action buttons: "Manual Entry", "View History", "Export Data"

#### Screen 5: Reports (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: Back arrow, "Reports"
- **Filters section:**
  - Type dropdown: Fuel▼
  - Period dropdown: 30d▼
  - Ships dropdown: All▼
  - "Generate" button
- **Quick Stats card:**
  - Voyages: 47
  - ETA Accuracy: 96.2%
  - Fuel consumed: 425.8KL
  - Efficiency: 2.1 L/nm
- **Consumption chart:**
  - Horizontal bar chart (simplified)
  - Top ships displayed
- Action buttons: "View Details", "Export PDF", "Export Excel", "Email Report"

#### Screen 6: Menu Sidebar (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: TytoAlba, Close button [X]
- **User profile:**
  - Avatar 👤
  - Role: Fleet Manager
  - Email: fleet@bahtera.co
- **Navigation menu (vertical):**
  - 🏠 Dashboard
  - 🚢 Ships
  - 🔮 ETA Prediction
  - ⛽ Fuel Monitor
  - 📊 Reports
  - 📈 Analytics
  - ⚙ Settings
- Divider
- 👥 Admin Panel (Admin only)
- Divider
- 🚪 Logout
- Footer: Version v1.2.3

#### Screen 7: Alerts (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: Hamburger menu, "Alerts (3)"
- **Alert cards (stacked):**
  - **Alert 1: ⚠ LOW FUEL**
    - Martha Baruna
    - Fuel: 7,500L, 15% remaining
    - 2 minutes ago
    - [View Details] [Dismiss]
  - **Alert 2: ℹ ROUTE UPDATE**
    - Rasuna Baruna
    - Weather advisory, Slight delay expected
    - 15 minutes ago
    - [View Details] [Dismiss]
  - **Alert 3: ⚙ SENSOR ISSUE**
    - Devi Baruna
    - FUEL-004 Offline 2 hrs
    - 1 hour ago
    - [Diagnose] [Dismiss]

#### Screen 8: Manual Fuel Entry (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- Header: Back arrow, "Manual Fuel Entry"
- Info text: "Use when sensor is offline or malfunctioning"
- **Form fields:**
  - Ship dropdown: Rasuna Baruna▼
  - Tank dropdown: Tank 1 (Main)▼
  - Fuel Level (L): Text input field
  - Reading Method dropdown: Visual Gauge▼
  - Notes: Textarea
- "Submit Reading" button
- "Cancel" link
- Footer note: "ℹ Manual entries require captain authentication"

#### Screen 9: Login (Mobile Web)
- **Browser address bar:** `https://tytoalba...`
- TytoAlba logo and "Fleet Monitor" tagline
- **Login form:**
  - Email input field
  - Password input field
  - "Remember me" checkbox
  - "LOGIN" button
  - "Forgot password?" link
- Footer: Version v1.2.3, PT Bahtera Adhiguna

**Mobile Web Design Principles:**
- **Browser-based:** Accessed via mobile browsers, not app stores
- **Responsive layout:** Same Vue.js app, adapted for 320px-767px screens
- **Touch-optimized:** 44x44px minimum touch targets
- **Vertical scrolling:** Single-column layouts
- **Simplified navigation:** Hamburger menu instead of sidebar
- **Compressed data:** Key info prioritized, details on tap
- **Address bar visible:** Unlike native apps
- **No app store:** Direct URL access (https://tytoalba.bahtera.co.id)
- **Progressive Web App (PWA) capable:** Can be installed as shortcut
- **Offline support:** Service workers for basic caching

**User Roles:** All roles (Captain, Fleet Manager, Administrator) - same web app, responsive design

**Comparison to Desktop:**
| Feature | Desktop | Mobile Web |
|---------|---------|------------|
| Navigation | Sidebar (240px) | Hamburger menu |
| Layout | Multi-column | Single column |
| Charts | Full-featured | Simplified |
| Tables | Full width | Horizontal scroll |
| Touch targets | Mouse precision | 44x44px minimum |
| Viewport | 1024px+ | 320px-767px |

---

## Design Specifications

### Color Scheme

**Status Colors:**
- Green (#10b981): Normal, Completed, Active, Online
- Blue (#3b82f6): In Transit, In Progress, Info
- Orange (#f59e0b): Warning, Degraded, Low Fuel
- Red (#ef4444): Critical, Delayed, Error, Offline
- Gray (#94a3b8): Inactive, Archived, Remaining Route

**UI Elements:**
- Primary: Blue (#3b82f6)
- Secondary: Gray (#6b7280)
- Background: White (#ffffff) / Light Gray (#f9fafb)
- Text: Dark Gray (#1f2937)
- Borders: Light Gray (#e5e7eb)

### Typography

**Desktop:**
- Headers: 24-32px, Bold
- Subheaders: 18-20px, Semibold
- Body: 14-16px, Regular
- Small text: 12px, Regular
- Code/Data: Monospace font

**Mobile:**
- Headers: 20-24px, Bold
- Subheaders: 16-18px, Semibold
- Body: 14px, Regular
- Small text: 11-12px, Regular

### Layout Grid

**Desktop:**
- Sidebar: 240px fixed width
- Main content: Fluid (responsive)
- Margins: 24px
- Card padding: 24px
- Table row height: 48px

**Mobile:**
- Single column layout
- Margins: 16px
- Card padding: 16px
- Touch targets: Minimum 44x44px

### Icons

Using emoji icons for wireframes (will be replaced with icon library in implementation):
- 🚢 Ship
- ⛵ Sailing
- ⚓ Port/Anchor
- 📍 Location
- ⏱ Time/Speed
- ⛽ Fuel
- 🔔 Notifications
- ⚙ Settings
- 👥 Users
- 📊 Reports/Charts
- 📈 Analytics
- 🔮 Prediction
- ⚠ Warning
- ℹ Info
- ✓ Success
- ✗ Error

**Implementation:** Use Lucide Icons or Heroicons for production.

---

## Responsive Breakpoints

**Desktop/Tablet:**
- Large Desktop: ≥1920px
- Desktop: 1280px - 1919px
- Laptop: 1024px - 1279px
- Tablet: 768px - 1023px

**Mobile:**
- Mobile Large: 414px - 767px
- Mobile Medium: 375px - 413px
- Mobile Small: 320px - 374px

---

## User Flows

### 1. Fleet Manager - Check Ship ETA
1. Login → Dashboard
2. View fleet map
3. Click ship in table → Ship Detail View
4. View ETA prediction with confidence
5. Review route on map
6. Check fuel estimate on arrival

### 2. Captain - Manual Fuel Entry (Mobile Web)
1. Open browser → Login (https://tytoalba.bahtera.co.id)
2. Tap hamburger menu → Fuel Monitor
3. Select ship from dropdown
4. Tap "Manual Entry"
5. Fill form: Tank, fuel level, reading method, notes
6. Tap "Submit Reading" → Confirmation

### 3. Fleet Manager - Generate Report
1. Dashboard → Reports
2. Select Report Type: "Fuel Consumption"
3. Select Time Period: "Last 30 Days"
4. Select Ships: All Ships
5. Click "Generate Report"
6. Review charts and tables
7. Click "Export Report (PDF)" or "Email Report"

### 4. Administrator - Add New User
1. Login → Admin Panel
2. Click "Manage Users"
3. Click "+ Add New User"
4. Fill form: Username, Email, Role, Ship Assignment, Permissions
5. Click "Create User"
6. System sends password email to new user

### 5. Captain - View Alert and Take Action (Mobile Web)
1. Open TytoAlba web app on mobile browser
2. See notification badge (🔔3) in header
3. Tap hamburger menu → Navigate to Alerts
4. View alert: "⚠ LOW FUEL - Martha Baruna - Fuel: 7,500L (15%)"
5. Tap "View Details" → Ship Detail screen with fuel highlighted
6. Review consumption rate and ETA
7. Decision: Continue to port or divert for refuel
8. Tap "Dismiss" on alert

---

## Accessibility Considerations

### WCAG 2.1 Level AA Compliance:

1. **Color Contrast:**
   - Text: Minimum 4.5:1 contrast ratio
   - Large text: Minimum 3:1 contrast ratio
   - Status indicators: Don't rely on color alone (use icons + text)

2. **Keyboard Navigation:**
   - All interactive elements keyboard accessible
   - Tab order logical and intuitive
   - Focus indicators visible

3. **Screen Reader Support:**
   - Semantic HTML elements
   - ARIA labels for icons and complex components
   - Alt text for images and charts

4. **Responsive Text:**
   - Text can be resized up to 200% without loss of functionality
   - Avoid fixed pixel sizes for text containers

5. **Touch Targets (Mobile):**
   - Minimum 44x44px for all tappable elements
   - Adequate spacing between buttons (8px minimum)

---

## Implementation Technologies

### Frontend:
- **Framework:** Vue.js 3 (Composition API) + TypeScript
- **UI Components:** Tailwind CSS + Headless UI
- **Charts:** Chart.js or Apache ECharts
- **Maps:** Leaflet.js with OpenStreetMap tiles
- **Icons:** Lucide Icons or Heroicons
- **Forms:** VeeValidate for validation

### Mobile Web (Responsive):
- **Same as Desktop:** Vue.js 3 responsive design
- **CSS Framework:** Tailwind CSS with responsive utilities
- **Media Queries:** Breakpoints at 768px, 640px, 480px
- **Touch Handling:** Native browser touch events
- **PWA:** Service workers for offline capability (optional)

---

## Exporting Wireframes

### For Presentations (PNG):
```bash
java -jar plantuml.jar -tpng -scale 2 *.puml
```

### For Web Documentation (SVG):
```bash
java -jar plantuml.jar -tsvg *.puml
```

### For Print Documents (PDF):
```bash
java -jar plantuml.jar -tpdf *.puml
```

---

## Mockup Tools (Next Phase)

For high-fidelity mockups after wireframe approval:
- **Figma:** Collaborative design (recommended)
- **Adobe XD:** Alternative design tool
- **Sketch:** macOS design tool
- **Penpot:** Open-source alternative

---

## Version History

- **v1.0 (2024-10-21):** Initial wireframes created for all 7 files
  - Desktop: Dashboard, Ship Detail, ETA Prediction, Fuel Monitor, Reports, Admin Panel (6 screens)
  - Mobile Web (Responsive): Dashboard, Ship Detail, ETA Prediction, Fuel Monitor, Reports, Menu Sidebar, Alerts, Manual Entry, Login (9 screens)

---

## Feedback & Iteration

### Review Checklist:
- [ ] All user roles can access their required features
- [ ] Navigation is intuitive and consistent
- [ ] Information hierarchy is clear
- [ ] Critical actions are easily accessible
- [ ] Error states and empty states are handled
- [ ] Loading states are considered
- [ ] Mobile responsive layouts work on all screen sizes
- [ ] Accessibility requirements are met

### Next Steps:
1. **Stakeholder Review:** Present wireframes to PT Bahtera Adhiguna stakeholders
2. **User Testing:** Conduct wireframe walkthroughs with captains and fleet managers
3. **Iterate:** Incorporate feedback and create v1.1
4. **High-Fidelity Mockups:** Move to Figma for detailed visual design
5. **Prototype:** Create interactive prototype for UAT
6. **Development:** Hand off to development team with design specifications

---

**Created for:** Tugas UTS/UAS - Arsitektur Perangkat Lunak untuk Digital Enterprise 2025
**Project:** TytoAlba - Predictive Monitoring System for Coal Delivery Vessel
**Team:** Kelompok 2 (Angga Pratama Suryabrata, Rina Widyasti Habibah, Putri Nur Meilisa)
