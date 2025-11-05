# TytoAlba UML Diagrams

This directory contains PlantUML source files for all system diagrams used in the TytoAlba project documentation.

## Diagram Files

1. **`usecase-diagram.puml`** - Use Case Diagram showing all system actors and their interactions
2. **`system-architecture.puml`** - System Architecture (C4 Container Diagram) showing deployment architecture
3. **`sequence-eta-prediction.puml`** - Sequence diagram for ETA prediction flow using LSTM
4. **`sequence-fuel-monitoring.puml`** - Sequence diagram for real-time fuel monitoring and consumption tracking
5. **`class-diagram.puml`** - Class diagram showing domain model and relationships

## How to Render Diagrams

### Method 1: Online PlantUML Editor (Easiest)

1. Go to [PlantUML Online Editor](http://www.plantuml.com/plantuml/uml/)
2. Copy the contents of any `.puml` file
3. Paste into the editor
4. The diagram will render automatically
5. Download as PNG/SVG using the download button

### Method 2: VS Code Extension (Recommended for Development)

1. Install the **PlantUML** extension by jebbs
   ```
   code --install-extension jebbs.plantuml
   ```

2. Install Graphviz (required dependency):
   - **Windows:** Download from [Graphviz website](https://graphviz.org/download/) or use Chocolatey:
     ```bash
     choco install graphviz
     ```
   - **macOS:**
     ```bash
     brew install graphviz
     ```
   - **Linux (Ubuntu/Debian):**
     ```bash
     sudo apt-get install graphviz
     ```

3. Open any `.puml` file in VS Code
4. Press `Alt+D` to preview the diagram
5. Right-click → "Export Current Diagram" to save as PNG/SVG/PDF

### Method 3: Command Line (For Batch Processing)

1. Install PlantUML JAR:
   ```bash
   # Download the latest version
   wget https://github.com/plantuml/plantuml/releases/download/v1.2023.13/plantuml-1.2023.13.jar
   ```

2. Render all diagrams:
   ```bash
   # Render as PNG
   java -jar plantuml.jar *.puml

   # Render as SVG (recommended for web)
   java -jar plantuml.jar -tsvg *.puml

   # Render as PDF
   java -jar plantuml.jar -tpdf *.puml
   ```

3. Output files will be created in the same directory

### Method 4: Docker (No Installation Required)

```bash
# Navigate to this directory
cd docs/diagrams

# Render all diagrams to PNG
docker run --rm -v $(pwd):/data plantuml/plantuml:latest -tpng /data/*.puml

# Render all diagrams to SVG
docker run --rm -v $(pwd):/data plantuml/plantuml:latest -tsvg /data/*.puml
```

### Method 5: IntelliJ IDEA / PyCharm

1. Install the **PlantUML integration** plugin
2. Open any `.puml` file
3. The diagram will render in a side panel
4. Right-click diagram → "Save Diagram" to export

## Diagram Descriptions

### 1. Use Case Diagram
**Purpose:** Shows all system actors (Captain, Fleet Manager, Admin, IoT Sensors) and their interactions with the system.

**Key Features Illustrated:**
- Fuel monitoring (automated and manual)
- ETA prediction using LSTM
- Route visualization
- Reporting and analytics
- System administration
- IoT data collection

**Actors:**
- Ship Captain (end user on vessel)
- Fleet Manager (office-based monitoring)
- System Administrator (technical management)
- Fuel Sensor (IoT device - automated actor)

### 2. System Architecture Diagram
**Purpose:** C4-style container diagram showing the complete technical architecture.

**Layers:**
- **Vessel Layer:** IoT sensors, Edge Gateway with MQTT client
- **Messaging Layer:** MQTT Broker (Eclipse Mosquitto)
- **Data Layer:** PostgreSQL + TimescaleDB, Redis cache
- **Application Layer:** Go Backend API, Python ML Service, Data Processor
- **ML Pipeline:** Feature Store, LSTM Model, MLflow Model Registry
- **Monitoring Layer:** Prometheus, Grafana, ELK Stack
- **Client Layer:** Vue.js 3 web application

**External Dependencies:**
- Weather API (OpenWeatherMap)
- AIS Data Source (Marine traffic)

### 3. Sequence Diagram - ETA Prediction Flow
**Purpose:** Detailed step-by-step flow for requesting and generating ETA predictions.

**Key Phases:**
1. **Request Phase:** Fleet manager initiates prediction
2. **Data Retrieval:** Fetch current ship data, historical AIS data, vessel specs
3. **ML Prediction:** Call external APIs (weather, AIS traffic), preprocess features
4. **Feature Engineering:** Normalize, calculate distance, encode temporal features
5. **Model Inference:** Bi-LSTM forward pass with attention mechanism
6. **Post-Processing:** Calculate confidence intervals, estimate fuel
7. **Storage & Response:** Log prediction, cache result, return to frontend
8. **Display Phase:** Update map visualization with ETA

**Total Steps:** 40+ numbered interactions

### 4. Sequence Diagram - Fuel Monitoring
**Purpose:** Real-time fuel monitoring from sensor reading to dashboard display.

**Key Scenarios:**
1. **Automated Sensor Reading** (every 5 minutes)
   - Sensor measures fuel level
   - Edge gateway buffers and publishes to MQTT
   - Data processor validates and stores in TimescaleDB
   - Alert service checks thresholds

2. **Real-time Dashboard View**
   - Captain opens dashboard
   - Backend fetches current fuel level
   - Frontend updates gauge visualization

3. **Fuel Consumption History**
   - Query 7-day historical data
   - Calculate consumption metrics
   - Render time-series chart

4. **Manual Fuel Entry** (fallback for sensor failure)
   - Captain enters reading manually
   - System validates and stores with source='MANUAL'

5. **Predictive Fuel Analysis**
   - Background task predicts fuel depletion
   - Alerts if refuel needed before arrival

**Total Steps:** 50+ numbered interactions

### 5. Class Diagram
**Purpose:** Complete domain model showing all entities and relationships.

**Core Entities:**
- **Ship** (central entity) - vessel with location, speed, status
- **Voyage** - trip from origin to destination port
- **Route** - sequence of waypoints
- **FuelReading** - time-series sensor/manual data
- **Prediction** - ML model outputs with confidence
- **WeatherForecast** - external weather data
- **Sensor** - IoT device metadata
- **User** - system users with roles
- **Alert** - notifications and warnings

**Key Relationships:**
- Ship has many FuelReadings, Sensors, Voyages, Predictions
- Voyage has one Route with many Waypoints
- Prediction has ConfidenceInterval, can be ETAPrediction subtype
- FuelReading comes from Sensor or User (manual)
- Alert triggered by Ship, acknowledged by User

**Design Patterns:**
- Inheritance: ETAPrediction extends Prediction
- Composition: Route contains Waypoints
- Aggregation: Ship associated with Port
- Enumerations: ShipStatus, VesselType, UserRole, etc.

## Customization

### Changing Colors/Styles

Edit the PlantUML files and modify:

```plantuml
' Change theme
!theme plain
' or
!theme aws-orange

' Change skin parameters
skinparam backgroundColor #FFFFFF
skinparam classBackgroundColor #LIGHTBLUE
```

### Adding New Diagrams

1. Create a new `.puml` file in this directory
2. Start with basic PlantUML structure:
   ```plantuml
   @startuml DiagramName
   ' Your diagram code here
   @enduml
   ```
3. Update this README with description
4. Render using one of the methods above

## Integration with Documentation

To include diagrams in Markdown files (like `Tugas UTS.md`):

### Option 1: Embed as Images (After Rendering)

```markdown
![Use Case Diagram](diagrams/usecase-diagram.png)
```

### Option 2: Link to PlantUML Server (Live Rendering)

```markdown
![Use Case Diagram](http://www.plantuml.com/plantuml/proxy?src=https://raw.githubusercontent.com/your-repo/TytoAlba/main/docs/diagrams/usecase-diagram.puml)
```

### Option 3: VS Code Markdown Preview (with PlantUML extension)

The diagrams will render automatically in Markdown preview if you have the PlantUML extension installed.

## Exporting for UAS Report

For the final UAS report, export diagrams as:

1. **PNG** (300 DPI) for Word/PDF documents:
   ```bash
   java -jar plantuml.jar -tpng -scale 2 *.puml
   ```

2. **SVG** for web-based reports (best quality):
   ```bash
   java -jar plantuml.jar -tsvg *.puml
   ```

3. **PDF** for direct inclusion in LaTeX documents:
   ```bash
   java -jar plantuml.jar -tpdf *.puml
   ```

## Troubleshooting

### "Graphviz not found" Error
- Install Graphviz (see Method 2 above)
- Add Graphviz bin directory to PATH
- Restart VS Code/terminal

### "Cannot render large diagram"
- Increase memory for Java:
  ```bash
  java -Xmx1024m -jar plantuml.jar diagram.puml
  ```

### "C4 theme not found"
- The system architecture diagram uses C4-PlantUML library
- It fetches from GitHub automatically
- Ensure internet connection for first render
- Alternatively, download C4 files locally

### Diagram Looks Messy
- Try different layout directions:
  ```plantuml
  left to right direction
  ' or
  top to bottom direction
  ```
- Adjust `skinparam linetype`:
  ```plantuml
  skinparam linetype ortho
  ' or
  skinparam linetype polyline
  ```

## Version History

- **v1.0 (2024-10-21):** Initial creation of all 5 core diagrams for UTS report

## References

- [PlantUML Official Documentation](https://plantuml.com/)
- [PlantUML Language Reference](https://plantuml.com/guide)
- [C4 Model](https://c4model.com/)
- [PlantUML for VS Code](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml)

---

**Created for:** Tugas UTS - Arsitektur Perangkat Lunak untuk Digital Enterprise 2025
**Project:** TytoAlba - Predictive Monitoring System for Coal Delivery Vessel
**Team:** Kelompok 2 (Angga, Rina, Putri)
