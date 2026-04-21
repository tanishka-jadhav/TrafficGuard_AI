from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Allow requests from Live Server (port 5500)

# ─── Synthetic Training Data ──────────────────────────────────────────────────
def generate_training_data(n=5000):
    np.random.seed(42)
    records = []
    for _ in range(n):
        hour = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        month = np.random.randint(1, 13)
        weather = np.random.choice(['Clear','Rain','Fog','Snow','Storm'], p=[0.5,0.25,0.1,0.1,0.05])
        traffic_density = np.random.randint(0, 101)
        speed_avg = np.random.randint(10, 100)
        road_type = np.random.choice(['Highway','Urban','Rural','Intersection'])
        visibility = np.random.randint(10, 101)

        risk = 0
        if hour in range(7,10) or hour in range(17,20): risk += 25
        if hour in range(0,4): risk += 20
        if weather == 'Storm': risk += 35
        elif weather == 'Snow': risk += 28
        elif weather == 'Fog': risk += 22
        elif weather == 'Rain': risk += 15
        if traffic_density > 75: risk += 20
        if speed_avg > 80: risk += 15
        if road_type == 'Intersection': risk += 20
        elif road_type == 'Highway': risk += 10
        if visibility < 30: risk += 20
        if day_of_week in [5,6]: risk += 8
        if month in [12,1,2]: risk += 10
        risk += np.random.randint(-10, 11)
        risk = max(0, min(100, risk))

        records.append({
            'hour': hour, 'day_of_week': day_of_week, 'month': month,
            'weather': weather, 'traffic_density': traffic_density,
            'speed_avg': speed_avg, 'road_type': road_type,
            'visibility': visibility, 'risk_score': risk,
            'accident': 1 if risk > 55 else 0
        })
    return pd.DataFrame(records)

# ─── Train ────────────────────────────────────────────────────────────────────
df = generate_training_data()
le_weather = LabelEncoder()
le_road = LabelEncoder()
df['weather_enc'] = le_weather.fit_transform(df['weather'])
df['road_enc'] = le_road.fit_transform(df['road_type'])

FEATURES = ['hour','day_of_week','month','weather_enc','traffic_density','speed_avg','road_enc','visibility']
X = df[FEATURES]

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, df['accident'])

reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
reg.fit(X, df['risk_score'])

print("✅ Models trained! API running at http://127.0.0.1:5000")

# ─── Helper ───────────────────────────────────────────────────────────────────
def predict_risk(hour, dow, month, weather, td, sp, road, vis):
    try:
        we = le_weather.transform([weather])[0]
        re = le_road.transform([road])[0]
    except:
        we, re = 0, 0
    feats = np.array([[hour, dow, month, we, td, sp, re, vis]])
    score = float(np.clip(reg.predict(feats)[0], 0, 100))
    prob  = float(clf.predict_proba(feats)[0][1])
    return score, prob

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.json
    try:
        hour = int(d.get('hour', 12))
        dow  = int(d.get('day_of_week', 1))
        mon  = int(d.get('month', 6))
        weather = d.get('weather', 'Clear')
        td   = int(d.get('traffic_density', 50))
        sp   = int(d.get('speed_avg', 50))
        road = d.get('road_type', 'Urban')
        vis  = int(d.get('visibility', 80))

        score, prob = predict_risk(hour, dow, mon, weather, td, sp, road, vis)

        if score < 30:   level, color = 'LOW',      '#00e676'
        elif score < 55: level, color = 'MODERATE',  '#ffd32a'
        elif score < 75: level, color = 'HIGH',      '#ff6b35'
        else:            level, color = 'CRITICAL',  '#ff1744'

        factors = []
        if 7<=hour<=9 or 17<=hour<=19: factors.append({'label':'Peak Rush Hour','impact':'high'})
        if 0<=hour<=3:                 factors.append({'label':'Late Night','impact':'high'})
        if weather in ['Storm','Snow','Fog']: factors.append({'label':f'Severe Weather: {weather}','impact':'critical'})
        elif weather == 'Rain':        factors.append({'label':'Rainy Conditions','impact':'medium'})
        if td > 75:   factors.append({'label':'High Traffic Density','impact':'high'})
        if sp > 80:   factors.append({'label':'High Speed','impact':'high'})
        if vis < 30:  factors.append({'label':'Low Visibility','impact':'critical'})
        if road == 'Intersection': factors.append({'label':'Complex Intersection','impact':'medium'})
        if not factors: factors.append({'label':'Conditions Nominal','impact':'low'})

        recs = {
            'LOW':      'Safe to drive. Stay alert and follow speed limits.',
            'MODERATE': 'Use caution. Reduce speed 10–15%, increase following distance.',
            'HIGH':     'Risk detected. Consider alternate routes or delay travel.',
            'CRITICAL': 'Dangerous! Avoid travel if possible. Emergency services alerted.'
        }
        return jsonify({
            'risk_score': round(score, 1),
            'accident_probability': round(prob * 100, 1),
            'risk_level': level,
            'risk_color': color,
            'contributing_factors': factors,
            'recommendation': recs[level]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/heatmap', methods=['GET'])
def heatmap():
    lat     = float(request.args.get('lat', 18.52))
    lng     = float(request.args.get('lng', 73.85))
    hour    = int(request.args.get('hour', 12))
    weather = request.args.get('weather', 'Clear')
    dow     = datetime.now().weekday()
    month   = datetime.now().month

    points = []
    grid = 20
    for i in range(grid):
        for j in range(grid):
            dlat = (i - grid//2) * 0.007
            dlng = (j - grid//2) * 0.007
            td  = random.randint(10, 100)
            sp  = random.randint(15, 100)
            vis = random.randint(15, 100)
            rt  = random.choice(['Highway','Urban','Rural','Intersection'])
            rs, _ = predict_risk(hour, dow, month, weather, td, sp, rt, vis)
            rs = float(np.clip(rs + random.uniform(-4, 4), 0, 100))
            points.append({'lat': lat+dlat, 'lng': lng+dlng, 'risk': round(rs,1)})
    return jsonify({'points': points})


@app.route('/api/timeseries', methods=['GET'])
def timeseries():
    weather = request.args.get('weather', 'Clear')
    road    = request.args.get('road_type', 'Urban')
    data = []
    for h in range(24):
        td  = 85 if (7<=h<=9 or 17<=h<=19) else 28
        sp  = 30 if (7<=h<=9 or 17<=h<=19) else 68
        vis = 95 if 6<=h<=18 else 50
        rs, prob = predict_risk(h, 1, 6, weather, td, sp, road, vis)
        data.append({'hour': h, 'label': f'{h:02d}:00', 'risk': round(rs,1), 'probability': round(prob*100,1)})
    return jsonify({'data': data})


@app.route('/api/hotspots', methods=['GET'])
def hotspots():
    lat = float(request.args.get('lat', 18.52))
    lng = float(request.args.get('lng', 73.85))
    spots = [
        {'name':'Main Junction',  'lat':lat+0.022,'lng':lng+0.016,'risk':82,'accidents_ytd':47},
        {'name':'Highway Merge',  'lat':lat-0.016,'lng':lng+0.032,'risk':76,'accidents_ytd':38},
        {'name':'School Zone',    'lat':lat+0.011,'lng':lng-0.021,'risk':68,'accidents_ytd':29},
        {'name':'Rail Crossing',  'lat':lat-0.026,'lng':lng-0.011,'risk':71,'accidents_ytd':33},
        {'name':'Market Circle',  'lat':lat+0.031,'lng':lng-0.026,'risk':63,'accidents_ytd':22},
    ]
    return jsonify({'hotspots': spots})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
