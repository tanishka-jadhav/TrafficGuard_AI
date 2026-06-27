# TrafficGuard AI

TrafficGuard AI is a smart-city traffic risk prediction demo that combines a Flask backend with a static frontend dashboard. It estimates road accident risk from contextual inputs such as time, weather, traffic density, speed, road type, and visibility, then visualizes the result through a tactical-style UI with maps, charts, and hotspot views.

This project is best understood as a prototype or academic/demo build rather than a production deployment. The machine learning models are trained on synthetic data generated at startup, and several frontend metrics are illustrative.

## What The Project Does

- Predicts a traffic risk score from 0 to 100
- Estimates accident probability as a percentage
- Classifies risk into `LOW`, `MODERATE`, `HIGH`, or `CRITICAL`
- Shows likely contributing factors for the predicted risk
- Displays a heatmap-style map overlay of nearby risk zones
- Shows hotspot markers and an hourly risk trend
- Includes a polished demo login/register interface

## Project Structure

```text
traffic_guard/
|-- backend/
|   |-- app.py
|   `-- requirements.txt
|-- frontend/
|   |-- index.html
|   `-- login.html
`-- README.md
```

## Tech Stack

### Backend

- Python
- Flask
- Flask-CORS
- NumPy
- pandas
- scikit-learn

### Frontend

- HTML, CSS, JavaScript
- Leaflet
- Leaflet.heat
- Chart.js
- OpenStreetMap tiles

## How It Works

1. The backend generates synthetic traffic-risk training data when the server starts.
2. Categorical fields such as weather and road type are encoded numerically.
3. Two ML models are trained:
   - `RandomForestClassifier` for accident/no-accident prediction
   - `GradientBoostingRegressor` for continuous risk score prediction
4. The frontend sends user-selected traffic conditions to the backend.
5. The backend returns:
   - risk score
   - accident probability
   - risk level
   - contributing factors
   - recommendation text
6. The frontend renders the result as charts, map overlays, and alert cards.

## Data Science Concepts Used

This project uses several core data science and machine learning concepts:

### 1. Synthetic Data Generation

The model is not trained on a real accident dataset. Instead, the backend programmatically creates thousands of synthetic records using randomized traffic, weather, visibility, seasonal, and time-based conditions.

Why it matters:

- useful for prototyping when real data is unavailable
- helps simulate many operating conditions quickly
- allows model training without building a full data pipeline first

### 2. Feature Engineering

The model uses structured input features that are relevant to traffic risk:

- `hour`
- `day_of_week`
- `month`
- `weather`
- `traffic_density`
- `speed_avg`
- `road_type`
- `visibility`

This is a classic feature-engineering step where domain knowledge is translated into model-ready variables.

### 3. Heuristic Labeling

The synthetic dataset creates a `risk_score` using rule-based logic:

- rush hour increases risk
- late night increases risk
- bad weather increases risk
- high speed and density increase risk
- low visibility increases risk
- intersections and winter months add risk

Then a binary target called `accident` is created from that score.

This is an example of heuristic target construction, where expert-style rules are used to generate labels.

### 4. Categorical Encoding

The project uses `LabelEncoder` to convert text categories such as weather and road type into numeric values that scikit-learn models can consume.

Examples:

- `Clear`, `Rain`, `Fog`, `Snow`, `Storm`
- `Highway`, `Urban`, `Rural`, `Intersection`

### 5. Supervised Learning

The backend trains on labeled examples and learns the relationship between traffic conditions and outcomes.

Two supervised learning tasks are used:

- classification: predict whether accident risk crosses a threshold
- regression: predict the numeric risk score itself

### 6. Ensemble Learning

Both models used are ensemble methods:

- `RandomForestClassifier`
- `GradientBoostingRegressor`

Ensemble models combine many weaker learners to produce stronger predictions, often improving robustness and non-linear pattern capture.

### 7. Probability Estimation

The classifier uses `predict_proba()` to estimate accident probability, not just a hard class label. This is important for risk systems because decision-makers often need confidence-style outputs.

### 8. Risk Stratification

The numeric score is converted into operational categories:

- `LOW`
- `MODERATE`
- `HIGH`
- `CRITICAL`

This is a practical analytics pattern where continuous model output is turned into actionable business tiers.

### 9. Time-Series Style Risk Profiling

The `/api/timeseries` route evaluates risk across the 24 hours of a day. While this is not a forecasting model, it does create a temporal risk profile that helps visualize how accident risk changes over time.

### 10. Geospatial Visualization

The frontend displays risk on a map using:

- heatmaps
- hotspot markers
- location-based overlays

This reflects a geospatial analytics concept, even though the map points are simulated rather than derived from a real GIS dataset.

## Current ML Inputs And Outputs

### Inputs

- hour of day
- day of week
- month
- weather condition
- traffic density
- average speed
- road type
- visibility

### Outputs

- risk score out of 100
- accident probability percentage
- risk level
- contributing factors
- driving recommendation

## API Endpoints

### `POST /api/predict`

Predicts traffic risk for a given input scenario.

Example request:

```json
{
  "hour": 18,
  "day_of_week": 2,
  "month": 6,
  "weather": "Rain",
  "traffic_density": 78,
  "speed_avg": 62,
  "road_type": "Intersection",
  "visibility": 40
}
```

Example response:

```json
{
  "risk_score": 67.4,
  "accident_probability": 71.2,
  "risk_level": "HIGH",
  "risk_color": "#ff6b35",
  "contributing_factors": [
    { "label": "Peak Rush Hour", "impact": "high" }
  ],
  "recommendation": "Risk detected. Consider alternate routes or delay travel."
}
```

### `GET /api/heatmap`

Returns simulated map points around a location with associated risk values.

Query params:

- `lat`
- `lng`
- `hour`
- `weather`

### `GET /api/timeseries`

Returns 24-hour risk and probability data for charting.

Query params:

- `weather`
- `road_type`

### `GET /api/hotspots`

Returns example hotspot markers near a selected location.

Query params:

- `lat`
- `lng`

## Setup Instructions

### 1. Clone The Project

```bash
git clone <your-repo-url>
cd traffic_guard
```

### 2. Create A Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

## Run The Project

### Start The Backend

From the project root:

```bash
python backend/app.py
```

The API will run at:

```text
http://127.0.0.1:5000
```

### Start The Frontend

Open the `frontend` folder with a local static server. For example:

```bash
cd frontend
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500/login.html
```

You can also use the VS Code Live Server extension if that is your preferred workflow.

## Demo Flow

1. Open `login.html`
2. Sign in or register using the frontend demo form
3. Enter the dashboard
4. Choose time, weather, road type, traffic density, speed, and visibility
5. Click `ANALYZE RISK`
6. Explore the heatmap, hotspots, and analytics panels

## Important Notes And Limitations

- The training data is synthetic, not collected from real traffic systems.
- The model is retrained every time the backend starts.
- The login/register flow is frontend-only and does not use a real auth backend.
- The hotspot list is hard-coded demo data.
- The heatmap points are simulated around the selected location.
- The accuracy number shown in the UI is decorative and not calculated in the backend.
- There is no database, user management, model persistence, or deployment config yet.

## Possible Future Improvements

- Train on real accident, weather, and road telemetry data
- Add proper train/test evaluation and report real metrics
- Replace `LabelEncoder` with a safer categorical pipeline for production
- Persist trained models with `joblib` or similar
- Add database-backed user accounts and saved analyses
- Use real GIS hotspot and incident data
- Add forecasting and anomaly detection for live traffic streams
- Deploy backend and frontend with environment-based configuration

## Summary

TrafficGuard AI is a visually strong ML demo for traffic accident risk assessment. It already demonstrates supervised learning, ensemble modeling, feature engineering, heuristic labeling, probability estimation, temporal analysis, and geospatial visualization in a single project. With real datasets and evaluation, it could evolve from a prototype into a much more credible smart-mobility analytics system.
