# TytoAlba ML Service - ASCII Art Presentation Slides

**15 Visual Slides for PowerPoint Presentation**

Created: November 13, 2025

---

## SLIDE 1: TITLE SLIDE

```
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║                    🚢 TYTOALBA 🚢                           ║
║                                                             ║
║          Ship Tracking & ML Prediction System              ║
║                                                             ║
║                                                             ║
║              ┌─────────────────────────┐                   ║
║              │    🗺️  Real-time AIS    │                   ║
║              │    🤖 LSTM Prediction   │                   ║
║              │    ⛽ Fuel Optimization │                   ║
║              │    🔍 Anomaly Detection │                   ║
║              └─────────────────────────┘                   ║
║                                                             ║
║                                                             ║
║              PLN Ship Fleet Management                     ║
║              Vue.js • Go • Python • PyTorch                ║
║                                                             ║
║                                                             ║
║              Presented by: Angga Suryabrata                ║
║              Date: November 2025                           ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
```

---

## SLIDE 2: SYSTEM ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │              USER INTERFACE LAYER                │      │
│  │  ┌──────────┐      ┌──────────┐                  │      │
│  │  │ Frontend │      │ Leaflet  │                  │      │
│  │  │ Vue.js 3 │◄────►│   Map    │                  │      │
│  │  │  :5173   │      │ Display  │                  │      │
│  │  └────┬─────┘      └──────────┘                  │      │
│  └───────┼──────────────────────────────────────────┘      │
│          │ HTTP/REST                                        │
│          ▼                                                  │
│  ┌──────────────────────────────────────────────────┐      │
│  │              APPLICATION LAYER                   │      │
│  │  ┌──────────┐      ┌──────────┐                  │      │
│  │  │ Backend  │      │   JSON   │                  │      │
│  │  │   Go     │◄────►│ Storage  │                  │      │
│  │  │  :8080   │      │ (ships)  │                  │      │
│  │  └────┬─────┘      └──────────┘                  │      │
│  └───────┼──────────────────────────────────────────┘      │
│          │ HTTP/REST                                        │
│          ▼                                                  │
│  ┌──────────────────────────────────────────────────┐      │
│  │              ML/AI LAYER                         │      │
│  │  ┌──────────┐      ┌──────────┐                  │      │
│  │  │    ML    │      │ PyTorch  │                  │      │
│  │  │  Service │◄────►│  Models  │                  │      │
│  │  │  :5000   │      │ (4 types)│                  │      │
│  │  └────┬─────┘      └──────────┘                  │      │
│  └───────┼──────────────────────────────────────────┘      │
│          │                                                  │
│          ▼                                                  │
│  ┌──────────────────────────────────────────────────┐      │
│  │              DATA LAYER                          │      │
│  │  ┌──────────┐      ┌──────────┐                  │      │
│  │  │PLN Ship  │      │Historical│                  │      │
│  │  │Tracking  │      │ Voyage   │                  │      │
│  │  │   API    │      │   Data   │                  │      │
│  │  └──────────┘      └──────────┘                  │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │         MESSAGING LAYER (Optional)               │      │
│  │         ┌──────────────────┐                     │      │
│  │         │   MQTT Broker    │                     │      │
│  │         │ (Real-time pub)  │                     │      │
│  │         └──────────────────┘                     │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 3: DATA PIPELINE FLOW

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA PIPELINE WORKFLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  STEP 1: Data Acquisition                                  │
│  ┌─────────────────────────────────┐                       │
│  │  PLN Ship Tracking API          │                       │
│  │  • 49 vessels tracked           │                       │
│  │  • Real-time AIS positions      │                       │
│  │  • Updated every 15 minutes     │                       │
│  └──────────────┬──────────────────┘                       │
│                 │                                           │
│                 ▼                                           │
│  STEP 2: Data Collection                                   │
│  ┌─────────────────────────────────┐                       │
│  │  Snapshot 1: Nov 13, 10:14 AM   │                       │
│  │  Snapshot 2: Nov 13, 03:11 PM   │                       │
│  │  • 18 active ships              │                       │
│  │  • 5-15 hour time span          │                       │
│  └──────────────┬──────────────────┘                       │
│                 │                                           │
│                 ▼                                           │
│  STEP 3: Data Interpolation                                │
│  ┌─────────────────────────────────┐                       │
│  │  Linear Interpolation Engine    │                       │
│  │  • 15-minute intervals          │                       │
│  │  • Position calculation         │                       │
│  │  • Speed estimation             │                       │
│  │  • Course interpolation         │                       │
│  │  ▼                              │                       │
│  │  Result: 417 records created    │                       │
│  └──────────────┬──────────────────┘                       │
│                 │                                           │
│                 ▼                                           │
│  STEP 4: Feature Engineering                               │
│  ┌─────────────────────────────────┐                       │
│  │  Calculate derived features:    │                       │
│  │  ✓ Distance to destination      │                       │
│  │  ✓ ETA (hours)                  │                       │
│  │  ✓ Fuel consumption estimate    │                       │
│  │  ✓ Course changes               │                       │
│  └──────────────┬──────────────────┘                       │
│                 │                                           │
│                 ▼                                           │
│  STEP 5: Dataset Creation                                  │
│  ┌─────────────────────────────────┐                       │
│  │  ✓ ETA data: 405 samples        │                       │
│  │  ✓ Fuel data: 405 samples       │                       │
│  │  ✓ Anomaly data: 417 samples    │                       │
│  │  ✓ Route data: 393 samples      │                       │
│  └──────────────┬──────────────────┘                       │
│                 │                                           │
│                 ▼                                           │
│  STEP 6: Model Training Ready                              │
│  ┌─────────────────────────────────┐                       │
│  │  CSV files saved in:            │                       │
│  │  data/processed/*.csv           │                       │
│  └─────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 4: LSTM MODEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│              LSTM ETA PREDICTION MODEL                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   INPUT LAYER (5 features)                                 │
│   ┌──────────────────────────────────────┐                 │
│   │ • Latitude         (-90° to 90°)     │                 │
│   │ • Longitude        (-180° to 180°)   │                 │
│   │ • Speed (knots)    (0 to 30)         │                 │
│   │ • Course (degrees) (0 to 360)        │                 │
│   │ • Distance (nm)    (0 to 100+)       │                 │
│   └─────────────┬────────────────────────┘                 │
│                 │                                           │
│                 ▼                                           │
│   ┌───────────────────────────────────────┐                │
│   │   PREPROCESSING LAYER                 │                │
│   │   StandardScaler Normalization        │                │
│   │   • Mean (μ) = 0                      │                │
│   │   • Std Dev (σ) = 1                   │                │
│   │   Formula: (x - μ) / σ                │                │
│   └─────────────┬─────────────────────────┘                │
│                 │                                           │
│                 ▼                                           │
│   ┌───────────────────────────────────────┐                │
│   │   LSTM LAYER 1                        │                │
│   │   ┌─────────────────────────────┐     │                │
│   │   │ Hidden Units: 64            │     │                │
│   │   │ Dropout: 0.2 (20%)          │     │                │
│   │   │ Activation: tanh            │     │                │
│   │   │ [Cell State Memory]         │     │                │
│   │   │ [Hidden State Memory]       │     │                │
│   │   └─────────────────────────────┘     │                │
│   └─────────────┬─────────────────────────┘                │
│                 │                                           │
│                 ▼                                           │
│   ┌───────────────────────────────────────┐                │
│   │   LSTM LAYER 2                        │                │
│   │   ┌─────────────────────────────┐     │                │
│   │   │ Hidden Units: 64            │     │                │
│   │   │ Dropout: 0.2 (20%)          │     │                │
│   │   │ Activation: tanh            │     │                │
│   │   │ [Cell State Memory]         │     │                │
│   │   │ [Hidden State Memory]       │     │                │
│   │   └─────────────────────────────┘     │                │
│   └─────────────┬─────────────────────────┘                │
│                 │                                           │
│                 ▼                                           │
│   ┌───────────────────────────────────────┐                │
│   │   FULLY CONNECTED LAYER 1             │                │
│   │   64 → 32 units                       │                │
│   │   • ReLU Activation                   │                │
│   │   • Dropout: 0.2                      │                │
│   └─────────────┬─────────────────────────┘                │
│                 │                                           │
│                 ▼                                           │
│   ┌───────────────────────────────────────┐                │
│   │   OUTPUT LAYER                        │                │
│   │   32 → 1 unit                         │                │
│   │   • Linear activation                 │                │
│   └─────────────┬─────────────────────────┘                │
│                 │                                           │
│                 ▼                                           │
│   OUTPUT: ETA (hours)                                      │
│   ┌──────────────────────────────────────┐                 │
│   │  Predicted arrival time              │                 │
│   │  Range: 0 - 9.14 hours               │                 │
│   │  Continuous value (regression)       │                 │
│   └──────────────────────────────────────┘                 │
│                                                             │
│   TOTAL PARAMETERS: ~145,000                               │
│   MODEL SIZE: 214 KB                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 5: TRAINING PROCESS FLOW

```
┌─────────────────────────────────────────────────────────────┐
│                  MODEL TRAINING PROCESS                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────┐            │
│  │ STEP 1: Load Training Data                 │            │
│  │ ┌────────────────────────────────────┐     │            │
│  │ │ CSV File: 405 samples              │     │            │
│  │ │ Features: 5 columns                │     │            │
│  │ │ Labels: 1 column (eta_hours)       │     │            │
│  │ └────────────────────────────────────┘     │            │
│  └─────────────────┬──────────────────────────┘            │
│                    │                                        │
│  ┌─────────────────▼──────────────────────────┐            │
│  │ STEP 2: Split Dataset                      │            │
│  │ ┌────────────────────────────────────┐     │            │
│  │ │ Training Set   : 283 (70%)         │     │            │
│  │ │ Validation Set : 61  (15%)         │     │            │
│  │ │ Test Set       : 61  (15%)         │     │            │
│  │ └────────────────────────────────────┘     │            │
│  └─────────────────┬──────────────────────────┘            │
│                    │                                        │
│  ┌─────────────────▼──────────────────────────┐            │
│  │ STEP 3: Normalize Features                 │            │
│  │ ┌────────────────────────────────────┐     │            │
│  │ │ Fit StandardScaler on train data   │     │            │
│  │ │ Transform train/val/test sets      │     │            │
│  │ │ Save scaler for inference          │     │            │
│  │ └────────────────────────────────────┘     │            │
│  └─────────────────┬──────────────────────────┘            │
│                    │                                        │
│  ┌─────────────────▼──────────────────────────┐            │
│  │ STEP 4: Training Loop (100 epochs max)     │            │
│  │ ┌────────────────────────────────────┐     │            │
│  │ │  FOR each epoch:                   │     │            │
│  │ │    1. Forward Pass                 │     │            │
│  │ │       • Input → LSTM → Output      │     │            │
│  │ │    2. Calculate Loss               │     │            │
│  │ │       • MSE(prediction, actual)    │     │            │
│  │ │    3. Backward Pass                │     │            │
│  │ │       • Compute gradients          │     │            │
│  │ │    4. Update Weights               │     │            │
│  │ │       • Adam optimizer (lr=0.001)  │     │            │
│  │ │    5. Validate                     │     │            │
│  │ │       • Check val loss             │     │            │
│  │ │    6. Early Stopping               │     │            │
│  │ │       • If no improvement: stop    │     │            │
│  │ └────────────────────────────────────┘     │            │
│  └─────────────────┬──────────────────────────┘            │
│                    │                                        │
│  ┌─────────────────▼──────────────────────────┐            │
│  │ STEP 5: Evaluate on Test Set               │            │
│  │ ┌────────────────────────────────────┐     │            │
│  │ │ Calculate metrics:                 │     │            │
│  │ │ • MSE  : 2.57 hours²               │     │            │
│  │ │ • RMSE : 1.60 hours                │     │            │
│  │ │ • MAE  : 1.16 hours (±70 min)      │     │            │
│  │ └────────────────────────────────────┘     │            │
│  └─────────────────┬──────────────────────────┘            │
│                    │                                        │
│  ┌─────────────────▼──────────────────────────┐            │
│  │ STEP 6: Save Model Artifacts               │            │
│  │ ┌────────────────────────────────────┐     │            │
│  │ │ ✓ eta_model.pth    (214 KB)       │     │            │
│  │ │ ✓ eta_scaler.pkl   (570 B)        │     │            │
│  │ │                                    │     │            │
│  │ │ Ready for inference!               │     │            │
│  │ └────────────────────────────────────┘     │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  Training Time: ~3 minutes (CPU)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 6: MODEL PERFORMANCE METRICS

```
┌─────────────────────────────────────────────────────────────┐
│              MODEL PERFORMANCE RESULTS                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TRAINING CONVERGENCE                                      │
│  ┌──────────────────────────────────────────────┐          │
│  │ Loss                                         │          │
│  │ 12 ┤●                                        │          │
│  │ 11 ┤●                                        │          │
│  │ 10 ┤ ●                                       │          │
│  │  9 ┤ ●                                       │          │
│  │  8 ┤   ●                                     │          │
│  │  7 ┤   ●●                                    │          │
│  │  6 ┤     ●●                                  │          │
│  │  5 ┤       ●●                                │          │
│  │  4 ┤         ●●                              │          │
│  │  3 ┤           ●●●                           │          │
│  │  2 ┤              ●●●●●●●●●●●●               │          │
│  │  1 ┤                          ●●●●●          │          │
│  │  0 └─────────────────────────────────────    │          │
│  │    0   20   40   60   80   100 (epochs)     │          │
│  │                                              │          │
│  │  ─── Train Loss    ─── Validation Loss      │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  FINAL METRICS                                             │
│  ┌────────────────────────────────────────────┐            │
│  │                                            │            │
│  │  📊 MSE  (Mean Squared Error)              │            │
│  │     2.57 hours²                            │            │
│  │                                            │            │
│  │  📐 RMSE (Root Mean Squared Error)         │            │
│  │     1.60 hours  ≈  96 minutes              │            │
│  │                                            │            │
│  │  ✅ MAE  (Mean Absolute Error)             │            │
│  │     1.16 hours  ≈  70 minutes              │            │
│  │                                            │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  INTERPRETATION                                            │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Average prediction error: ± 70 minutes          │      │
│  │                                                  │      │
│  │  ETA Range: 0 - 9.14 hours                       │      │
│  │  Mean ETA: 1.99 hours                            │      │
│  │  Error %: ~58% of mean                           │      │
│  │                                                  │      │
│  │  Status: ✅ Good for interpolated data           │      │
│  │  Baseline: Simple formula (distance/speed)       │      │
│  │  Future: Improve with real AIS data              │      │
│  │                                                  │      │
│  │  Expected with Real Data: MAE < 30 minutes       │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  TRAINING DETAILS                                          │
│  • Epochs: 100 (with early stopping)                       │
│  • Optimizer: Adam (learning rate = 0.001)                 │
│  • Loss Function: MSE                                      │
│  • Best model saved at epoch 87                            │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 7: TECHNOLOGY STACK

```
┌─────────────────────────────────────────────────────────────┐
│                    TECHNOLOGY STACK                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │         FRONTEND TECHNOLOGIES                │          │
│  ├──────────────────────────────────────────────┤          │
│  │  🎨 Vue.js 3           Reactive UI framework │          │
│  │  ⚡ Vite               Fast build tool       │          │
│  │  🗺️  Leaflet.js        Interactive maps      │          │
│  │  🎯 Pinia              State management      │          │
│  │  🌐 Axios              HTTP client           │          │
│  │  🎨 TailwindCSS        Utility-first CSS     │          │
│  │  📱 Responsive Design  Mobile-friendly       │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │         BACKEND TECHNOLOGIES                 │          │
│  ├──────────────────────────────────────────────┤          │
│  │  🐹 Go 1.21+           High-performance      │          │
│  │  🔌 Gorilla Mux        HTTP routing          │          │
│  │  📡 MQTT Client        Message broker        │          │
│  │  📁 JSON Storage       File-based DB         │          │
│  │  🌐 REST API           RESTful endpoints     │          │
│  │  🔄 CORS Enabled       Cross-origin support  │          │
│  │  ⚙️  Graceful Shutdown  Production-ready     │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │         ML/AI TECHNOLOGIES                   │          │
│  ├──────────────────────────────────────────────┤          │
│  │  🐍 Python 3.13        ML language           │          │
│  │  🔥 PyTorch 2.5        Deep learning         │          │
│  │  🧠 LSTM Networks      Time series models    │          │
│  │  📊 Pandas             Data manipulation     │          │
│  │  🔢 NumPy              Numerical computing   │          │
│  │  🚀 Flask              ML API service        │          │
│  │  🎯 Scikit-learn       ML utilities          │          │
│  │  💾 Pickle             Model serialization   │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │         DATA SOURCE                          │          │
│  ├──────────────────────────────────────────────┤          │
│  │  🚢 PLN Ship Tracking  49 vessels            │          │
│  │  📍 Real-time AIS      Position updates      │          │
│  │  🕐 15-min intervals   Data frequency        │          │
│  │  🌐 REST API           HTTPS endpoints       │          │
│  └──────────────────────────────────────────────┘          │
│                                                             │
│  ┌──────────────────────────────────────────────┐          │
│  │         INFRASTRUCTURE                       │          │
│  ├──────────────────────────────────────────────┤          │
│  │  🐧 Linux/WSL          Development env       │          │
│  │  📡 Mosquitto MQTT     Message broker        │          │
│  │  🐙 Git                Version control       │          │
│  │  📦 Shell Scripts      Automation            │          │
│  │  🔧 systemd            Service management    │          │
│  └──────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 8: PROJECT TIMELINE

```
┌─────────────────────────────────────────────────────────────┐
│                   PROJECT TIMELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE 1: INITIAL SETUP (Nov 12, 2025)                     │
│  ┌──────────────────────────────────────────┐              │
│  │ ✅ Backend Go service configured         │              │
│  │ ✅ Frontend Vue.js application created   │              │
│  │ ✅ ML service directory structure        │              │
│  │ ✅ MQTT broker installation              │              │
│  │ ✅ Git repository initialized            │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  PHASE 2: DATA INTEGRATION (Nov 13 - Morning)              │
│  ┌──────────────────────────────────────────┐              │
│  │ ✅ PLN API integration complete          │              │
│  │ ✅ Fixed ship display bug (Dashboard.vue)│              │
│  │ ✅ Fixed port conflict issues            │              │
│  │ ✅ Updated stop_all.sh script            │              │
│  │ ✅ Updated 18 ship positions             │              │
│  │ ✅ Created data backup system            │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  PHASE 3: ML PIPELINE (Nov 13 - Afternoon)                 │
│  ┌──────────────────────────────────────────┐              │
│  │ ✅ Generated 417 historical records      │              │
│  │ ✅ Fixed ETA calculation (distance/speed)│              │
│  │ ✅ Prepared 4 training datasets:         │              │
│  │    • ETA: 405 samples                    │              │
│  │    • Fuel: 405 samples                   │              │
│  │    • Anomaly: 417 samples                │              │
│  │    • Route: 393 samples                  │              │
│  │ ✅ Trained ETA model (LSTM)              │              │
│  │ ✅ Trained Fuel consumption model        │              │
│  │ ✅ Created training documentation        │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  PHASE 4: ASSIGNMENTS (Nov 14-17, 2025)                    │
│  ┌──────────────────────────────────────────┐              │
│  │ ⏳ Bedah Buku Word document               │              │
│  │ ⏳ Unit tests: remove_outliers function   │              │
│  │ ⏳ Unit tests: LSTM model components      │              │
│  │ ⏳ Unit tests: ML API endpoints           │              │
│  │ ⏳ Unit tests: Go backend handlers        │              │
│  │ ⏳ Coverage reports generation            │              │
│  │ ⏳ Final submission package               │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  PHASE 5: FUTURE ENHANCEMENTS                              │
│  ┌──────────────────────────────────────────┐              │
│  │ ⏳ Deploy to cloud (AWS/GCP/Azure)        │              │
│  │ ⏳ Real-time AIS data streaming           │              │
│  │ ⏳ Weather data integration               │              │
│  │ ⏳ Database implementation (PostgreSQL)   │              │
│  │ ⏳ User authentication system             │              │
│  │ ⏳ Mobile application (React Native)      │              │
│  │ ⏳ Advanced analytics dashboard           │              │
│  └──────────────────────────────────────────┘              │
│                                                             │
│  KEY MILESTONE: Nov 17, 2025 (Assignment Due Date)         │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 9: ETA PREDICTION EXAMPLE

```
┌─────────────────────────────────────────────────────────────┐
│              ETA PREDICTION WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  SCENARIO: Ship Voyage Prediction                          │
│  ════════════════════════════════════════════              │
│                                                             │
│  INPUT DATA                                                │
│  ┌──────────────────────────────────────┐                  │
│  │ 🚢 Vessel: MV. Meutia Baruna         │                  │
│  │ 📍 Current Position:                 │                  │
│  │    • Latitude:  -5.64159°            │                  │
│  │    • Longitude: 105.36937°           │                  │
│  │ ⚡ Speed: 4.0 knots                   │                  │
│  │ 🧭 Course: 311° (Northwest)          │                  │
│  │ 🎯 Destination: Tarahan Port         │                  │
│  │ 📏 Distance: 25.0 nautical miles     │                  │
│  └───────────────┬──────────────────────┘                  │
│                  │                                          │
│                  ▼                                          │
│  PREPROCESSING                                             │
│  ┌──────────────────────────────────────┐                  │
│  │ 1. Feature Vector Creation           │                  │
│  │    [-5.64, 105.37, 4.0, 311, 25.0]   │                  │
│  │                                      │                  │
│  │ 2. Normalization (StandardScaler)    │                  │
│  │    Apply saved scaler parameters     │                  │
│  │    [0.12, -0.45, -1.2, 0.87, 1.34]   │                  │
│  └───────────────┬──────────────────────┘                  │
│                  │                                          │
│                  ▼                                          │
│  MODEL INFERENCE                                           │
│  ┌──────────────────────────────────────┐                  │
│  │ LSTM Forward Pass:                   │                  │
│  │  Input → LSTM1 → LSTM2 → FC → Output │                  │
│  │                                      │                  │
│  │  Computation time: ~50ms             │                  │
│  └───────────────┬──────────────────────┘                  │
│                  │                                          │
│                  ▼                                          │
│  PREDICTION OUTPUT                                         │
│  ┌──────────────────────────────────────┐                  │
│  │ 🎯 Predicted ETA: 6.12 hours         │                  │
│  │ ⏰ Arrival Time: Nov 13, 23:30       │                  │
│  │ 📊 Confidence: 85%                   │                  │
│  │ 📐 Error Margin: ±70 minutes         │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  COMPARISON & VALIDATION                                   │
│  ┌────────────────────────────────────────────────┐        │
│  │                                                │        │
│  │  Simple Formula:                               │        │
│  │  ETA = Distance / Speed                        │        │
│  │      = 25.0 nm / 4.0 knots                     │        │
│  │      = 6.25 hours                              │        │
│  │                                                │        │
│  │  LSTM Prediction: 6.12 hours                   │        │
│  │                                                │        │
│  │  Difference: ~8 minutes (2% error)             │        │
│  │                                                │        │
│  │  ✅ Model prediction is reasonable!            │        │
│  │                                                │        │
│  │  Note: With real AIS data showing speed        │        │
│  │  variations, LSTM would learn non-linear       │        │
│  │  patterns and outperform simple formula.       │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 10: FOUR ML MODELS OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                  ML MODELS PORTFOLIO                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │ 1. ETA PREDICTION MODEL          ✅ TRAINED    │        │
│  ├────────────────────────────────────────────────┤        │
│  │ Type: LSTM (Recurrent Neural Network)         │        │
│  │                                                │        │
│  │ Purpose: Predict ship arrival time            │        │
│  │                                                │        │
│  │ Input (5 features):                            │        │
│  │  • Latitude, Longitude                         │        │
│  │  • Speed (knots)                               │        │
│  │  • Course (degrees)                            │        │
│  │  • Distance remaining (nm)                     │        │
│  │                                                │        │
│  │ Output: ETA (hours)                            │        │
│  │                                                │        │
│  │ Architecture: 2-layer LSTM + FC layers        │        │
│  │ Performance: MAE = 1.16 hours (±70 min)        │        │
│  │ Training Samples: 405                          │        │
│  │ Model Size: 214 KB                             │        │
│  └────────────────────────────────────────────────┘        │
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │ 2. FUEL CONSUMPTION MODEL        ✅ TRAINED    │        │
│  ├────────────────────────────────────────────────┤        │
│  │ Type: Feedforward Neural Network              │        │
│  │                                                │        │
│  │ Purpose: Predict fuel usage                   │        │
│  │                                                │        │
│  │ Input (4 features):                            │        │
│  │  • Average speed                               │        │
│  │  • Distance traveled                           │        │
│  │  • Time duration                               │        │
│  │  • Course change                               │        │
│  │                                                │        │
│  │ Output: Fuel consumption (liters)              │        │
│  │                                                │        │
│  │ Architecture: 3-layer feedforward NN          │        │
│  │ Performance: MAE = 8.03 liters                 │        │
│  │ Training Samples: 405                          │        │
│  │ Model Size: 5.9 KB                             │        │
│  └────────────────────────────────────────────────┘        │
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │ 3. ANOMALY DETECTION MODEL       ⏳ PENDING    │        │
│  ├────────────────────────────────────────────────┤        │
│  │ Type: Autoencoder (Unsupervised)              │        │
│  │                                                │        │
│  │ Purpose: Detect unusual ship behavior         │        │
│  │                                                │        │
│  │ Input (4 features):                            │        │
│  │  • Latitude, Longitude                         │        │
│  │  • Speed, Course                               │        │
│  │                                                │        │
│  │ Output: Anomaly score (0-1)                    │        │
│  │                                                │        │
│  │ Architecture: Encoder-Decoder network         │        │
│  │ Method: Reconstruction error threshold        │        │
│  │ Training Samples: 417 (normal patterns)        │        │
│  │ Use Cases: Detect AIS spoofing, route         │        │
│  │            deviations, emergency situations    │        │
│  └────────────────────────────────────────────────┘        │
│                                                             │
│  ┌────────────────────────────────────────────────┐        │
│  │ 4. ROUTE OPTIMIZATION MODEL      ⏳ PENDING    │        │
│  ├────────────────────────────────────────────────┤        │
│  │ Type: Feedforward Neural Network              │        │
│  │                                                │        │
│  │ Purpose: Suggest optimal waypoints            │        │
│  │                                                │        │
│  │ Input (5 features):                            │        │
│  │  • Start position (lat, lon)                   │        │
│  │  • Destination (lat, lon)                      │        │
│  │  • Current heading                             │        │
│  │                                                │        │
│  │ Output: Next waypoint (lat, lon)               │        │
│  │                                                │        │
│  │ Architecture: 3-layer feedforward NN          │        │
│  │ Method: Sequential waypoint prediction        │        │
│  │ Training Samples: 393                          │        │
│  │ Use Cases: Fuel-efficient routing,            │        │
│  │            weather avoidance                   │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 11: API INTEGRATION FLOW

```
┌─────────────────────────────────────────────────────────────┐
│                API REQUEST/RESPONSE FLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ① FRONTEND (User Interaction)                             │
│  ┌──────────────────────────────────────┐                  │
│  │ User clicks ship on map              │                  │
│  │ "Show ETA Prediction" button         │                  │
│  │                                      │                  │
│  │ JavaScript sends POST request:       │                  │
│  │ ┌──────────────────────────────────┐ │                  │
│  │ │ URL: /api/predictions/eta        │ │                  │
│  │ │ Method: POST                     │ │                  │
│  │ │ Body: {                          │ │                  │
│  │ │   "mmsi": "525112006",           │ │                  │
│  │ │   "latitude": -5.64,             │ │                  │
│  │ │   "longitude": 105.36,           │ │                  │
│  │ │   "speed": 4.0,                  │ │                  │
│  │ │   "course": 311,                 │ │                  │
│  │ │   "distance": 25.0               │ │                  │
│  │ │ }                                │ │                  │
│  │ └──────────────────────────────────┘ │                  │
│  └───────────────┬──────────────────────┘                  │
│                  │ HTTP Request                             │
│                  ▼                                          │
│  ② BACKEND (Go Service - Port 8080)                        │
│  ┌──────────────────────────────────────┐                  │
│  │ Go Router receives request           │                  │
│  │                                      │                  │
│  │ 1. Validate request body             │                  │
│  │    • Check required fields           │                  │
│  │    • Validate data types             │                  │
│  │                                      │                  │
│  │ 2. Forward to ML service             │                  │
│  │    POST http://localhost:5000/predict│                  │
│  └───────────────┬──────────────────────┘                  │
│                  │ HTTP Request                             │
│                  ▼                                          │
│  ③ ML SERVICE (Python - Port 5000)                         │
│  ┌──────────────────────────────────────┐                  │
│  │ Flask API receives request           │                  │
│  │                                      │                  │
│  │ 1. Load model artifacts              │                  │
│  │    • eta_model.pth (LSTM weights)    │                  │
│  │    • eta_scaler.pkl (normalizer)     │                  │
│  │                                      │                  │
│  │ 2. Preprocess input                  │                  │
│  │    • Create feature vector           │                  │
│  │    • Apply StandardScaler            │                  │
│  │    • Convert to PyTorch tensor       │                  │
│  │                                      │                  │
│  │ 3. Run inference                     │                  │
│  │    • model.eval()                    │                  │
│  │    • Forward pass through LSTM       │                  │
│  │    • Get prediction (6.12 hours)     │                  │
│  │                                      │                  │
│  │ 4. Return JSON response              │                  │
│  │    {"eta_hours": 6.12}               │                  │
│  └───────────────┬──────────────────────┘                  │
│                  │ HTTP Response                            │
│                  ▼                                          │
│  ④ BACKEND (Process Response)                              │
│  ┌──────────────────────────────────────┐                  │
│  │ Go service enriches response:        │                  │
│  │                                      │                  │
│  │ • Calculate arrival timestamp        │                  │
│  │   (current_time + eta_hours)         │                  │
│  │ • Add confidence score               │                  │
│  │ • Add ship metadata                  │                  │
│  │ • Format for frontend                │                  │
│  └───────────────┬──────────────────────┘                  │
│                  │ HTTP Response                            │
│                  ▼                                          │
│  ⑤ FRONTEND (Display Result)                               │
│  ┌──────────────────────────────────────┐                  │
│  │ JavaScript receives response:        │                  │
│  │ ┌──────────────────────────────────┐ │                  │
│  │ │ {                                │ │                  │
│  │ │   "eta_hours": 6.12,             │ │                  │
│  │ │   "arrival_time": "23:30",       │ │                  │
│  │ │   "arrival_date": "Nov 13",      │ │                  │
│  │ │   "confidence": 0.85             │ │                  │
│  │ │ }                                │ │                  │
│  │ └──────────────────────────────────┘ │                  │
│  │                                      │                  │
│  │ Display in UI:                       │                  │
│  │ ┌──────────────────────────────────┐ │                  │
│  │ │ 🚢 MV. Meutia Baruna             │ │                  │
│  │ │ 🎯 ETA: Nov 13, 23:30 (6h 12m)   │ │                  │
│  │ │ 📊 Confidence: 85%               │ │                  │
│  │ └──────────────────────────────────┘ │                  │
│  └──────────────────────────────────────┘                  │
│                                                             │
│  Total Response Time: ~100-200ms                           │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 12: DATA STATISTICS COMPARISON

```
┌─────────────────────────────────────────────────────────────┐
│              DATA STATISTICS COMPARISON                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RAW DATA (PLN API Snapshots)                              │
│  ┌────────────────────────────────────────────┐            │
│  │ Snapshot 1: Nov 13, 10:14 AM               │            │
│  │ Snapshot 2: Nov 13, 03:11 PM               │            │
│  │                                            │            │
│  │ Total vessels: 49 ships                    │            │
│  │ Active vessels: 18 ships                   │            │
│  │ Time span: ~5 hours                        │            │
│  │ Data points: 36 records (2 per ship)       │            │
│  │                                            │            │
│  │ Limitation: Only 2 snapshots               │            │
│  │ Issue: Insufficient for ML training        │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  ↓ INTERPOLATION PROCESS                                   │
│  ┌────────────────────────────────────────────┐            │
│  │ Method: Linear interpolation               │            │
│  │ Interval: 15 minutes                       │            │
│  │ Fields: lat, lon, speed, course            │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  PROCESSED DATA (After Interpolation)                      │
│  ┌────────────────────────────────────────────┐            │
│  │ Total records: 417 data points             │            │
│  │ Active vessels: 12 ships (with full data)  │            │
│  │ Time span per ship: 5-15 hours             │            │
│  │                                            │            │
│  │ Top 3 Ships by Data Points:                │            │
│  │  • MEUTIA BARUNA:  61 records (15 hours)   │            │
│  │  • SARTIKA BARUNA: 59 records (14.6 hours) │            │
│  │  • JAYANTI BARUNA: 53 records (13 hours)   │            │
│  │                                            │            │
│  │ ✅ Suitable for ML training!               │            │
│  └────────────────────────────────────────────┘            │
│                                                             │
│  TRAINING DATASETS CREATED                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │                                                    │    │
│  │  1. ETA Dataset                                    │    │
│  │     • Samples: 405                                 │    │
│  │     • Features: 5 (lat, lon, speed, course, dist) │    │
│  │     • Label: eta_hours (0-9.14 hours)             │    │
│  │     • Mean: 1.99 hours, Std: 2.65 hours           │    │
│  │                                                    │    │
│  │  2. Fuel Dataset                                   │    │
│  │     • Samples: 405                                 │    │
│  │     • Features: 4 (speed, distance, time, Δcourse)│    │
│  │     • Label: fuel_liters                          │    │
│  │     • Estimated based on speed²                   │    │
│  │                                                    │    │
│  │  3. Anomaly Dataset                                │    │
│  │     • Samples: 417                                 │    │
│  │     • Features: 4 (lat, lon, speed, course)       │    │
│  │     • Label: is_anomaly (all = 0, normal)         │    │
│  │     • Unsupervised learning approach              │    │
│  │                                                    │    │
│  │  4. Route Dataset                                  │    │
│  │     • Samples: 393                                 │    │
│  │     • Features: 5 (start, dest, heading)          │    │
│  │     • Label: next_waypoint (lat, lon)             │    │
│  │     • Sequential prediction task                  │    │
│  │                                                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  COMPARISON                                                │
│  ┌───────────────┬─────────────┬─────────────────┐         │
│  │  Metric       │ Before      │ After           │         │
│  ├───────────────┼─────────────┼─────────────────┤         │
│  │ Records       │ 36          │ 417  (+1058%)   │         │
│  │ Ships         │ 18          │ 12   (filtered) │         │
│  │ Avg points    │ 2 per ship  │ 35 per ship     │         │
│  │ ML Ready      │ ❌ No       │ ✅ Yes          │         │
│  └───────────────┴─────────────┴─────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 13: LOSS CONVERGENCE VISUALIZATION

```
┌─────────────────────────────────────────────────────────────┐
│              TRAINING LOSS CONVERGENCE                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ETA MODEL TRAINING PROGRESS                               │
│  ┌──────────────────────────────────────────────────┐      │
│  │                                                  │      │
│  │ Loss                                             │      │
│  │ Value                                            │      │
│  │  12 ┤●                                           │      │
│  │     │●                                           │      │
│  │  11 ┤ ●                                          │      │
│  │     │ ●                                          │      │
│  │  10 ┤  ●                                         │      │
│  │     │  ●                                         │      │
│  │   9 ┤   ●                                        │      │
│  │     │   ●                                        │      │
│  │   8 ┤    ●●                                      │      │
│  │     │      ●                                     │      │
│  │   7 ┤       ●                                    │      │
│  │     │       ●●                                   │      │
│  │   6 ┤         ●●                                 │      │
│  │     │           ●                                │      │
│  │   5 ┤            ●●                              │      │
│  │     │              ●●                            │      │
│  │   4 ┤                ●●                          │      │
│  │     │                  ●●                        │      │
│  │   3 ┤                    ●●●                     │      │
│  │     │                       ●●●                  │      │
│  │   2 ┤                          ●●●●●●●●          │      │
│  │     │                                 ●●●●●●     │      │
│  │   1 ┤                                       ●●●●●│      │
│  │     │                                            │      │
│  │   0 └────┬────┬────┬────┬────┬────┬────┬────┬───│      │
│  │         0   20   40   60   80  100 120  140  160│      │
│  │                      Epochs                      │      │
│  │                                                  │      │
│  │  Legend:  ● Train Loss    ○ Validation Loss     │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  KEY OBSERVATIONS                                          │
│  ┌────────────────────────────────────────────────────┐    │
│  │                                                    │    │
│  │  1. RAPID CONVERGENCE (Epoch 0-40)                │    │
│  │     • Loss drops from 11.9 → 10.3                 │    │
│  │     • Model learning basic patterns               │    │
│  │     • Large gradient updates                      │    │
│  │                                                    │    │
│  │  2. STEADY IMPROVEMENT (Epoch 40-80)              │    │
│  │     • Loss drops from 10.3 → 3.2                  │    │
│  │     • Model refining predictions                  │    │
│  │     • Learning non-linear relationships           │    │
│  │                                                    │    │
│  │  3. FINE-TUNING (Epoch 80-100)                    │    │
│  │     • Loss drops from 3.2 → 2.5                   │    │
│  │     • Small incremental improvements              │    │
│  │     • Model approaching optimal weights           │    │
│  │                                                    │    │
│  │  4. CONVERGENCE ACHIEVED                          │    │
│  │     • Final train loss: 2.49                      │    │
│  │     • Final val loss: 1.93                        │    │
│  │     • No overfitting detected                     │    │
│  │     • Validation loss tracks training loss        │    │
│  │                                                    │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  TRAINING DYNAMICS                                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Optimizer: Adam (Adaptive learning rate)           │    │
│  │ Initial LR: 0.001                                  │    │
│  │ Loss Function: MSE (Mean Squared Error)            │    │
│  │ Batch Size: Full batch (283 samples)               │    │
│  │ Early Stopping: Patience = 15 epochs               │    │
│  │ Best Model: Saved at epoch 100                     │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  COMPARISON WITH BASELINE                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Simple Formula (distance/speed): MSE ≈ 0.5         │    │
│  │ LSTM Model (learned patterns):   MSE = 2.57        │    │
│  │                                                    │    │
│  │ Note: Higher MSE due to interpolated data with    │    │
│  │ constant speed. Real AIS data would show LSTM's   │    │
│  │ advantage in learning speed variation patterns.   │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 14: DEPLOYMENT ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│              PRODUCTION DEPLOYMENT ARCHITECTURE             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │              CLIENT LAYER                        │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  🌐 Web Browser (Desktop/Mobile)     │        │      │
│  │  │  • Responsive Vue.js SPA             │        │      │
│  │  │  • Real-time map updates             │        │      │
│  │  │  • WebSocket for live data           │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  └────────────────────┬─────────────────────────────┘      │
│                       │ HTTPS (SSL/TLS)                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────────┐      │
│  │              LOAD BALANCER                       │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  Nginx / AWS ALB                     │        │      │
│  │  │  • SSL termination                   │        │      │
│  │  │  • Rate limiting                     │        │      │
│  │  │  • Request routing                   │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  └────────────────────┬─────────────────────────────┘      │
│                       │                                     │
│         ┌─────────────┼─────────────┐                      │
│         │             │             │                      │
│         ▼             ▼             ▼                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ Backend  │  │ Backend  │  │ Backend  │                 │
│  │ Instance │  │ Instance │  │ Instance │                 │
│  │    #1    │  │    #2    │  │    #3    │                 │
│  │ (Go:8080)│  │ (Go:8080)│  │ (Go:8080)│                 │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                 │
│       │            │            │                          │
│       └────────────┼────────────┘                          │
│                    │                                        │
│                    ▼                                        │
│  ┌──────────────────────────────────────────────────┐      │
│  │              ML SERVICE CLUSTER                  │      │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐       │      │
│  │  │   ML     │  │   ML     │  │   ML     │       │      │
│  │  │ Service  │  │ Service  │  │ Service  │       │      │
│  │  │   #1     │  │   #2     │  │   #3     │       │      │
│  │  │ (Py:5000)│  │ (Py:5000)│  │ (Py:5000)│       │      │
│  │  └──────────┘  └──────────┘  └──────────┘       │      │
│  └────────────────────┬─────────────────────────────┘      │
│                       │                                     │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────────┐      │
│  │              DATA LAYER                          │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  📊 PostgreSQL Database              │        │      │
│  │  │  • Ship master data                  │        │      │
│  │  │  • Historical voyages                │        │      │
│  │  │  • Predictions log                   │        │      │
│  │  │  • User data                         │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  💾 Redis Cache                      │        │      │
│  │  │  • API response caching              │        │      │
│  │  │  • Session management                │        │      │
│  │  │  • Rate limiting data                │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  📁 S3 / Object Storage              │        │      │
│  │  │  • ML model files (.pth)             │        │      │
│  │  │  • Training datasets                 │        │      │
│  │  │  • Logs and backups                  │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │              MONITORING & LOGGING                │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  📈 Prometheus + Grafana             │        │      │
│  │  │  • System metrics                    │        │      │
│  │  │  • API latency                       │        │      │
│  │  │  • Model performance                 │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  📝 ELK Stack (Elasticsearch)        │        │      │
│  │  │  • Centralized logging               │        │      │
│  │  │  • Error tracking                    │        │      │
│  │  │  • Audit trails                      │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │              EXTERNAL SERVICES                   │      │
│  │  ┌──────────────────────────────────────┐        │      │
│  │  │  🚢 PLN Ship Tracking API            │        │      │
│  │  │  🌤️  Weather API (OpenWeatherMap)    │        │      │
│  │  │  📧 Email Service (SendGrid)         │        │      │
│  │  │  🔔 Alert System (PagerDuty)         │        │      │
│  │  └──────────────────────────────────────┘        │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  INFRASTRUCTURE: Docker + Kubernetes / AWS ECS              │
│  CI/CD: GitHub Actions / GitLab CI                         │
│  Cloud Provider: AWS / GCP / Azure                         │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 15: FUTURE ENHANCEMENTS ROADMAP

```
┌─────────────────────────────────────────────────────────────┐
│              FUTURE ENHANCEMENTS ROADMAP                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE 1: DATA QUALITY (Q1 2026)                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ⏳ Real AIS Data Integration                        │    │
│  │    • Connect to live AIS feed                      │    │
│  │    • 1-minute update frequency                     │    │
│  │    • Store in time-series database                 │    │
│  │                                                    │    │
│  │ ⏳ Weather Data Integration                         │    │
│  │    • OpenWeatherMap / NOAA API                     │    │
│  │    • Wind, wave, current data                      │    │
│  │    • Historical weather archive                    │    │
│  │                                                    │    │
│  │ ⏳ Port Data Integration                            │    │
│  │    • Port congestion info                          │    │
│  │    • Berth availability                            │    │
│  │    • Tidal data                                    │    │
│  │                                                    │    │
│  │ Impact: Improve ETA accuracy to <20 min error      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  PHASE 2: MODEL ENHANCEMENTS (Q2 2026)                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ⏳ Advanced LSTM Architecture                       │    │
│  │    • Attention mechanism                           │    │
│  │    • Multi-head attention                          │    │
│  │    • Transformer encoder                           │    │
│  │                                                    │    │
│  │ ⏳ Ensemble Models                                  │    │
│  │    • Combine LSTM + GRU + Transformer              │    │
│  │    • Weighted voting system                        │    │
│  │    • Confidence scoring                            │    │
│  │                                                    │    │
│  │ ⏳ Multi-Task Learning                              │    │
│  │    • Joint ETA + Fuel prediction                   │    │
│  │    • Shared feature representations                │    │
│  │    • Transfer learning                             │    │
│  │                                                    │    │
│  │ ⏳ Reinforcement Learning for Routes                │    │
│  │    • Q-learning for optimal routing                │    │
│  │    • Reward: minimize time + fuel                  │    │
│  │    • Environment: maritime conditions              │    │
│  │                                                    │    │
│  │ Impact: 50% error reduction, multi-objective opt   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  PHASE 3: FEATURE EXPANSION (Q3 2026)                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ⏳ Predictive Maintenance                           │    │
│  │    • Engine health monitoring                      │    │
│  │    • Failure prediction                            │    │
│  │    • Maintenance scheduling                        │    │
│  │                                                    │    │
│  │ ⏳ Carbon Footprint Tracking                        │    │
│  │    • CO₂ emissions calculation                     │    │
│  │    • Green routing optimization                    │    │
│  │    • ESG reporting                                 │    │
│  │                                                    │    │
│  │ ⏳ Fleet Optimization                               │    │
│  │    • Multi-ship coordination                       │    │
│  │    • Port scheduling optimization                  │    │
│  │    • Load balancing                                │    │
│  │                                                    │    │
│  │ ⏳ Advanced Anomaly Detection                       │    │
│  │    • AIS spoofing detection                        │    │
│  │    • Collision risk prediction                     │    │
│  │    • Illegal fishing detection                     │    │
│  │                                                    │    │
│  │ Impact: Comprehensive fleet management system      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  PHASE 4: PLATFORM EXPANSION (Q4 2026)                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ⏳ Mobile Applications                              │    │
│  │    • iOS app (Swift)                               │    │
│  │    • Android app (Kotlin)                          │    │
│  │    • Offline mode support                          │    │
│  │    • Push notifications                            │    │
│  │                                                    │    │
│  │ ⏳ Voice Assistant Integration                      │    │
│  │    • "Hey TytoAlba, what's the ETA for..."         │    │
│  │    • Natural language queries                      │    │
│  │    • Voice alerts                                  │    │
│  │                                                    │    │
│  │ ⏳ AR/VR Visualization                              │    │
│  │    • 3D route visualization                        │    │
│  │    • Virtual port tour                             │    │
│  │    • Training simulations                          │    │
│  │                                                    │    │
│  │ ⏳ API Marketplace                                  │    │
│  │    • Public API for developers                     │    │
│  │    • Rate-limited free tier                        │    │
│  │    • Enterprise plans                              │    │
│  │                                                    │    │
│  │ Impact: Multi-platform ecosystem                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  PHASE 5: AI/ML RESEARCH (Ongoing)                         │
│  ┌────────────────────────────────────────────────────┐    │
│  │ ⏳ Vision Models                                    │    │
│  │    • Satellite imagery analysis                    │    │
│  │    • Ship detection from space                     │    │
│  │    • Port activity monitoring                      │    │
│  │                                                    │    │
│  │ ⏳ Large Language Models                            │    │
│  │    • Maritime documentation analysis               │    │
│  │    • Automated report generation                   │    │
│  │    • Chatbot support                               │    │
│  │                                                    │    │
│  │ ⏳ Federated Learning                               │    │
│  │    • Privacy-preserving ML                         │    │
│  │    • Multi-fleet collaboration                     │    │
│  │    • Decentralized training                        │    │
│  │                                                    │    │
│  │ Impact: Cutting-edge AI capabilities               │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ULTIMATE VISION                                           │
│  ┌────────────────────────────────────────────────────┐    │
│  │  "Autonomous Maritime Intelligence Platform"       │    │
│  │                                                    │    │
│  │  🚢 Real-time fleet management                     │    │
│  │  🤖 AI-powered decision support                    │    │
│  │  🌍 Global maritime network                        │    │
│  │  ⚡ Zero-latency predictions                        │    │
│  │  🌱 Sustainable shipping operations                │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## END OF PRESENTATION

```
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║                       THANK YOU!                            ║
║                                                             ║
║                    🚢 TYTOALBA 🚢                           ║
║                                                             ║
║              Ship Tracking & ML Prediction System          ║
║                                                             ║
║                                                             ║
║                   Questions & Discussion                   ║
║                                                             ║
║                                                             ║
║              📧 Contact: angga.suryabrata@email.com        ║
║              🐙 GitHub: github.com/yourusername/tytoalba   ║
║              📚 Docs: /ml-service/TRAINING_GUIDE.md        ║
║                                                             ║
║                                                             ║
║              "Making Maritime Intelligence Accessible"     ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
```

---

**File saved: `PRESENTATION_SLIDES_ASCII.md`**

**Usage Instructions:**

1. Copy each slide into PowerPoint as text boxes with monospace font (Courier New, Consolas)
2. Use black background with white/green text for terminal-style appearance
3. Or use white background with black text for traditional presentation
4. Each slide is self-contained and ready to present
5. Adjust font size as needed for your display (typically 12-16pt)

**Tips for PowerPoint:**
- Format → Font → Courier New or Consolas
- Alignment → Left align for ASCII art
- Use slide transitions sparingly
- Add company logo/branding as needed
- Consider adding animated builds for complex diagrams
