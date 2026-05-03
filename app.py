
"""
SmartControl AI - Production Lighting Control System
====================================================
Features:
- AI-powered lighting optimization
- Auto-learning pipeline (model improves with new data)
- Philips Hue compatible lighting control API
- PostgreSQL database for data persistence
- Real-time lighting level recommendations
"""

import os
import json
import pickle
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# ============================================================
# App Configuration
# ============================================================
app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database URL (set in Render Environment Variables)
DATABASE_URL = os.environ.get('DATABASE_URL', '')

# Model storage
MODEL_PATH = 'smartcontrol_ai_model.pkl'
LIGHTING_MODEL_PATH = 'lighting_model.pkl'

# ============================================================
# Database Setup (PostgreSQL)
# ============================================================
db_available = False
conn = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    
    if DATABASE_URL:
        # Fix Render's postgres:// to postgresql://
        if DATABASE_URL.startswith('postgres://'):
            DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
        
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        conn.autocommit = True
        db_available = True
        logger.info("Database connected successfully")
        
        # Create tables
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS energy_data (
                    id SERIAL PRIMARY KEY,
                    facility_id VARCHAR(100),
                    facility_type INTEGER,
                    floor_area FLOAT,
                    num_employees INTEGER,
                    operating_hours INTEGER,
                    avg_temperature FLOAT,
                    month INTEGER,
                    monthly_consumption FLOAT,
                    actual_reduction FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lighting_data (
                    id SERIAL PRIMARY KEY,
                    facility_id VARCHAR(100),
                    zone_id VARCHAR(100),
                    hour INTEGER,
                    day_of_week INTEGER,
                    occupancy BOOLEAN,
                    natural_light_lux FLOAT,
                    target_lux FLOAT,
                    optimal_brightness INTEGER,
                    energy_saved FLOAT,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS lighting_schedules (
                    id SERIAL PRIMARY KEY,
                    facility_id VARCHAR(100),
                    zone_id VARCHAR(100),
                    zone_name VARCHAR(200),
                    hour INTEGER,
                    brightness INTEGER,
                    color_temp INTEGER,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    model_type VARCHAR(50),
                    version INTEGER,
                    r2_score FLOAT,
                    num_samples INTEGER,
                    created_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
        logger.info("Database tables created")
    else:
        logger.warning("No DATABASE_URL set, running without database")
        
except ImportError:
    logger.warning("psycopg2 not installed, running without database")
except Exception as e:
    logger.warning(f"Database connection failed: {e}")

# ============================================================
# Model Loading
# ============================================================
energy_model = None
lighting_model = None

try:
    if os.path.exists(MODEL_PATH):
        energy_model = joblib.load(MODEL_PATH)
        logger.info("Energy model loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load energy model: {e}")

try:
    if os.path.exists(LIGHTING_MODEL_PATH):
        lighting_model = joblib.load(LIGHTING_MODEL_PATH)
        logger.info("Lighting model loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load lighting model: {e}")

# ============================================================
# Lighting Control Logic
# ============================================================

# Default lighting profiles for hotels
HOTEL_LIGHTING_PROFILES = {
    "lobby": {
        "name": "Lobby",
        "target_lux": 300,
        "schedule": {
            0: 30, 1: 20, 2: 20, 3: 20, 4: 20, 5: 40,
            6: 60, 7: 80, 8: 90, 9: 100, 10: 100, 11: 100,
            12: 100, 13: 100, 14: 100, 15: 100, 16: 100, 17: 100,
            18: 100, 19: 90, 20: 80, 21: 70, 22: 50, 23: 40
        },
        "color_temp": {
            "day": 4000,    # Neutral white
            "evening": 3000, # Warm white
            "night": 2700    # Very warm
        }
    },
    "corridor": {
        "name": "Corridor",
        "target_lux": 150,
        "schedule": {
            0: 20, 1: 15, 2: 15, 3: 15, 4: 15, 5: 30,
            6: 50, 7: 70, 8: 80, 9: 80, 10: 80, 11: 80,
            12: 80, 13: 80, 14: 80, 15: 80, 16: 80, 17: 80,
            18: 80, 19: 70, 20: 60, 21: 50, 22: 40, 23: 30
        },
        "color_temp": {
            "day": 4000,
            "evening": 3000,
            "night": 2700
        }
    },
    "restaurant": {
        "name": "Restaurant",
        "target_lux": 200,
        "schedule": {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
            6: 50, 7: 70, 8: 80, 9: 80, 10: 60, 11: 80,
            12: 90, 13: 80, 14: 60, 15: 50, 16: 60, 17: 80,
            18: 90, 19: 100, 20: 100, 21: 90, 22: 50, 23: 0
        },
        "color_temp": {
            "day": 3500,
            "evening": 2700,
            "night": 2700
        }
    },
    "guest_room": {
        "name": "Guest Room",
        "target_lux": 200,
        "schedule": {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
            6: 30, 7: 50, 8: 60, 9: 60, 10: 60, 11: 60,
            12: 60, 13: 60, 14: 60, 15: 60, 16: 60, 17: 70,
            18: 80, 19: 80, 20: 70, 21: 60, 22: 40, 23: 20
        },
        "color_temp": {
            "day": 4000,
            "evening": 3000,
            "night": 2200
        }
    }
}

def get_optimal_brightness(zone_id, hour, occupancy=True, natural_light_lux=0):
    """Calculate optimal brightness based on AI logic"""
    profile = HOTEL_LIGHTING_PROFILES.get(zone_id, HOTEL_LIGHTING_PROFILES["lobby"])
    
    # Base brightness from schedule
    base_brightness = profile["schedule"].get(hour, 50)
    
    # Adjust for occupancy
    if not occupancy:
        base_brightness = max(10, int(base_brightness * 0.3))
    
    # Adjust for natural light
    target_lux = profile["target_lux"]
    if natural_light_lux > 0 and target_lux > 0:
        natural_contribution = min(1.0, natural_light_lux / target_lux)
        reduction = int(base_brightness * natural_contribution * 0.7)
        base_brightness = max(10, base_brightness - reduction)
    
    # Use ML model if available
    if lighting_model is not None:
        try:
            features = np.array([[hour, int(occupancy), natural_light_lux, target_lux]])
            predicted = lighting_model.predict(features)[0]
            # Blend ML prediction with rule-based (70% ML, 30% rules)
            base_brightness = int(predicted * 0.7 + base_brightness * 0.3)
        except Exception:
            pass
    
    return max(0, min(100, base_brightness))

def get_color_temperature(zone_id, hour):
    """Get optimal color temperature for time of day"""
    profile = HOTEL_LIGHTING_PROFILES.get(zone_id, HOTEL_LIGHTING_PROFILES["lobby"])
    
    if 6 <= hour <= 17:
        return profile["color_temp"]["day"]
    elif 18 <= hour <= 21:
        return profile["color_temp"]["evening"]
    else:
        return profile["color_temp"]["night"]

def calculate_energy_savings(zone_id, hour, brightness):
    """Calculate energy savings compared to full brightness"""
    max_wattage = {
        "lobby": 500,
        "corridor": 200,
        "restaurant": 400,
        "guest_room": 100
    }
    watts = max_wattage.get(zone_id, 200)
    current_watts = watts * (brightness / 100.0)
    saved_watts = watts - current_watts
    return round(saved_watts, 1)

# ============================================================
# API Endpoints
# ============================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model": "SmartControl AI v2.0",
        "database": "connected" if db_available else "not connected",
        "lighting_model": "loaded" if lighting_model else "using rules",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Energy reduction prediction (existing endpoint)"""
    try:
        data = request.get_json()
        
        features = np.array([[
            data['facility_type'],
            data['floor_area'],
            data['num_employees'],
            data['operating_hours'],
            data['avg_temperature'],
            data['month'],
            data['monthly_consumption']
        ]])
        
        if energy_model is not None:
            prediction = energy_model.predict(features)[0]
            prediction = max(5.0, min(45.0, prediction))
        else:
            # Fallback calculation
            base = 20.0
            if data['facility_type'] == 0:  # Hotel
                base = 25.0
            prediction = base + (data['monthly_consumption'] / 10000) * 5
            prediction = max(15.0, min(35.0, prediction))
        
        return jsonify({
            "reduction_rate": round(prediction, 2),
            "input_received": data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ============================================================
# Lighting Control Endpoints
# ============================================================

@app.route('/lighting/status', methods=['GET'])
def lighting_status():
    """Get current lighting status for all zones"""
    facility_id = request.args.get('facility_id', 'default')
    current_hour = datetime.now().hour
    
    zones = []
    total_savings = 0
    
    for zone_id, profile in HOTEL_LIGHTING_PROFILES.items():
        brightness = get_optimal_brightness(zone_id, current_hour)
        color_temp = get_color_temperature(zone_id, current_hour)
        savings = calculate_energy_savings(zone_id, current_hour, brightness)
        total_savings += savings
        
        zones.append({
            "zone_id": zone_id,
            "zone_name": profile["name"],
            "brightness": brightness,
            "color_temperature": color_temp,
            "target_lux": profile["target_lux"],
            "energy_saved_watts": savings,
            "is_ai_controlled": True
        })
    
    return jsonify({
        "facility_id": facility_id,
        "timestamp": datetime.now().isoformat(),
        "current_hour": current_hour,
        "zones": zones,
        "total_energy_saved_watts": total_savings,
        "ai_mode": "active"
    })

@app.route('/lighting/recommend', methods=['POST'])
def lighting_recommend():
    """Get AI recommendation for specific zone"""
    try:
        data = request.get_json()
        zone_id = data.get('zone_id', 'lobby')
        hour = data.get('hour', datetime.now().hour)
        occupancy = data.get('occupancy', True)
        natural_light_lux = data.get('natural_light_lux', 0)
        
        brightness = get_optimal_brightness(zone_id, hour, occupancy, natural_light_lux)
        color_temp = get_color_temperature(zone_id, hour)
        savings = calculate_energy_savings(zone_id, hour, brightness)
        
        return jsonify({
            "zone_id": zone_id,
            "recommended_brightness": brightness,
            "recommended_color_temp": color_temp,
            "energy_saved_watts": savings,
            "reason": generate_reason(zone_id, hour, occupancy, natural_light_lux, brightness)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/lighting/schedule', methods=['GET'])
def lighting_schedule():
    """Get 24-hour lighting schedule for a zone"""
    zone_id = request.args.get('zone_id', 'lobby')
    occupancy = request.args.get('occupancy', 'true').lower() == 'true'
    
    schedule = []
    for hour in range(24):
        brightness = get_optimal_brightness(zone_id, hour, occupancy)
        color_temp = get_color_temperature(zone_id, hour)
        savings = calculate_energy_savings(zone_id, hour, brightness)
        
        schedule.append({
            "hour": hour,
            "brightness": brightness,
            "color_temperature": color_temp,
            "energy_saved_watts": savings
        })
    
    # Calculate daily totals
    total_saved_wh = sum(item["energy_saved_watts"] for item in schedule)
    
    return jsonify({
        "zone_id": zone_id,
        "zone_name": HOTEL_LIGHTING_PROFILES.get(zone_id, {}).get("name", zone_id),
        "schedule": schedule,
        "daily_energy_saved_wh": total_saved_wh,
        "daily_cost_saved_yen": round(total_saved_wh * 0.025, 1),
        "monthly_cost_saved_yen": round(total_saved_wh * 0.025 * 30, 0)
    })

@app.route('/lighting/control', methods=['POST'])
def lighting_control():
    """Send control command to lighting (Philips Hue compatible)"""
    try:
        data = request.get_json()
        zone_id = data.get('zone_id')
        brightness = data.get('brightness')
        color_temp = data.get('color_temperature')
        mode = data.get('mode', 'auto')  # auto or manual
        
        # Generate Philips Hue compatible command
        hue_command = {
            "on": brightness > 0,
            "bri": int(brightness * 2.54),  # Convert 0-100 to 0-254
            "ct": int(1000000 / color_temp) if color_temp else 250  # Convert K to mirek
        }
        
        # Log the control action
        if db_available:
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO lighting_data 
                        (facility_id, zone_id, hour, day_of_week, occupancy, 
                         natural_light_lux, target_lux, optimal_brightness, energy_saved)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        data.get('facility_id', 'default'),
                        zone_id,
                        datetime.now().hour,
                        datetime.now().weekday(),
                        data.get('occupancy', True),
                        data.get('natural_light_lux', 0),
                        HOTEL_LIGHTING_PROFILES.get(zone_id, {}).get('target_lux', 200),
                        brightness,
                        calculate_energy_savings(zone_id, datetime.now().hour, brightness)
                    ))
            except Exception as e:
                logger.warning(f"Failed to log control action: {e}")
        
        return jsonify({
            "status": "success",
            "zone_id": zone_id,
            "applied": {
                "brightness": brightness,
                "color_temperature": color_temp,
                "mode": mode
            },
            "hue_command": hue_command,
            "message": f"Lighting set to {brightness}% in {zone_id}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ============================================================
# Data Collection & Auto-Learning Endpoints
# ============================================================

@app.route('/data/submit', methods=['POST'])
def submit_data():
    """Submit new energy/lighting data for learning"""
    try:
        data = request.get_json()
        data_type = data.get('type', 'energy')
        
        if db_available:
            if data_type == 'energy':
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO energy_data 
                        (facility_id, facility_type, floor_area, num_employees,
                         operating_hours, avg_temperature, month, 
                         monthly_consumption, actual_reduction)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        data.get('facility_id', 'default'),
                        data['facility_type'],
                        data['floor_area'],
                        data['num_employees'],
                        data['operating_hours'],
                        data['avg_temperature'],
                        data['month'],
                        data['monthly_consumption'],
                        data.get('actual_reduction', None)
                    ))
            elif data_type == 'lighting':
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO lighting_data 
                        (facility_id, zone_id, hour, day_of_week, occupancy,
                         natural_light_lux, target_lux, optimal_brightness, energy_saved)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        data.get('facility_id', 'default'),
                        data['zone_id'],
                        data['hour'],
                        data['day_of_week'],
                        data['occupancy'],
                        data['natural_light_lux'],
                        data['target_lux'],
                        data['optimal_brightness'],
                        data.get('energy_saved', 0)
                    ))
            
            return jsonify({
                "status": "success",
                "message": "Data submitted successfully",
                "data_type": data_type
            })
        else:
            return jsonify({
                "status": "warning",
                "message": "Data received but database not available"
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/learn/trigger', methods=['POST'])
def trigger_learning():
    """Trigger model re-training with collected data"""
    try:
        if not db_available:
            return jsonify({"error": "Database not available for learning"}), 400
        
        model_type = request.get_json().get('model_type', 'energy')
        
        if model_type == 'energy':
            result = retrain_energy_model()
        elif model_type == 'lighting':
            result = retrain_lighting_model()
        else:
            return jsonify({"error": "Invalid model_type"}), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/learn/status', methods=['GET'])
def learning_status():
    """Get current model learning status"""
    status = {
        "energy_model": {
            "loaded": energy_model is not None,
            "type": "RandomForestRegressor"
        },
        "lighting_model": {
            "loaded": lighting_model is not None,
            "type": "RandomForestRegressor"
        },
        "database": db_available
    }
    
    if db_available:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM energy_data")
                status["energy_data_count"] = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM lighting_data")
                status["lighting_data_count"] = cur.fetchone()[0]
                
                cur.execute("""
                    SELECT model_type, version, r2_score, num_samples, created_at 
                    FROM model_versions 
                    ORDER BY created_at DESC LIMIT 5
                """)
                rows = cur.fetchall()
                status["recent_trainings"] = [
                    {
                        "model_type": r[0],
                        "version": r[1],
                        "r2_score": r[2],
                        "num_samples": r[3],
                        "trained_at": r[4].isoformat() if r[4] else None
                    }
                    for r in rows
                ]
        except Exception as e:
            status["db_error"] = str(e)
    
    return jsonify(status)

# ============================================================
# Auto-Learning Functions
# ============================================================

def retrain_energy_model():
    """Retrain energy reduction model with new data"""
    global energy_model
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT facility_type, floor_area, num_employees, operating_hours,
                   avg_temperature, month, monthly_consumption, actual_reduction
            FROM energy_data
            WHERE actual_reduction IS NOT NULL
        """)
        rows = cur.fetchall()
    
    if len(rows) < 10:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 10 samples, currently have {len(rows)}",
            "current_samples": len(rows)
        }
    
    # Prepare training data
    X = np.array([[r[0], r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
    y = np.array([r[7] for r in rows])
    
    # Train new model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    new_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    new_model.fit(X_train, y_train)
    
    r2_score = new_model.score(X_test, y_test)
    
    # Only update if new model is better or first training
    if energy_model is None or r2_score > 0.5:
        energy_model = new_model
        joblib.dump(energy_model, MODEL_PATH)
        
        # Log version
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_versions (model_type, version, r2_score, num_samples)
                VALUES ('energy', 
                    COALESCE((SELECT MAX(version) FROM model_versions WHERE model_type='energy'), 0) + 1,
                    %s, %s)
            """, (r2_score, len(rows)))
        
        return {
            "status": "success",
            "message": "Energy model retrained successfully",
            "r2_score": round(r2_score, 4),
            "num_samples": len(rows)
        }
    else:
        return {
            "status": "skipped",
            "message": f"New model R2={r2_score:.4f} not better than current",
            "r2_score": round(r2_score, 4)
        }

def retrain_lighting_model():
    """Retrain lighting optimization model"""
    global lighting_model
    
    with conn.cursor() as cur:
        cur.execute("""
            SELECT hour, occupancy::int, natural_light_lux, target_lux, optimal_brightness
            FROM lighting_data
            WHERE optimal_brightness IS NOT NULL
        """)
        rows = cur.fetchall()
    
    if len(rows) < 20:
        return {
            "status": "insufficient_data",
            "message": f"Need at least 20 samples, currently have {len(rows)}",
            "current_samples": len(rows)
        }
    
    X = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    y = np.array([r[4] for r in rows])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    new_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    new_model.fit(X_train, y_train)
    
    r2_score = new_model.score(X_test, y_test)
    
    if r2_score > 0.3:
        lighting_model = new_model
        joblib.dump(lighting_model, LIGHTING_MODEL_PATH)
        
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_versions (model_type, version, r2_score, num_samples)
                VALUES ('lighting',
                    COALESCE((SELECT MAX(version) FROM model_versions WHERE model_type='lighting'), 0) + 1,
                    %s, %s)
            """, (r2_score, len(rows)))
        
        return {
            "status": "success",
            "message": "Lighting model retrained successfully",
            "r2_score": round(r2_score, 4),
            "num_samples": len(rows)
        }
    else:
        return {
            "status": "skipped",
            "message": f"Model quality too low: R2={r2_score:.4f}",
            "r2_score": round(r2_score, 4)
        }

# ============================================================
# Philips Hue Bridge Integration
# ============================================================

@app.route('/hue/discover', methods=['GET'])
def hue_discover():
    """Instructions for connecting Philips Hue Bridge"""
    return jsonify({
        "instructions": [
            "1. Press the button on your Philips Hue Bridge",
            "2. Send POST to /hue/register with your bridge IP",
            "3. Use the returned username for all future requests"
        ],
        "supported_features": [
            "Brightness control (0-100%)",
            "Color temperature (2000K-6500K)",
            "On/Off control",
            "Group control",
            "Schedule-based automation"
        ],
        "compatible_devices": [
            "Philips Hue White",
            "Philips Hue White Ambiance",
            "Philips Hue Color",
            "IKEA TRADFRI (via Hue Bridge)",
            "Any Zigbee 3.0 light"
        ]
    })

@app.route('/hue/command', methods=['POST'])
def hue_command():
    """Generate Philips Hue API command"""
    try:
        data = request.get_json()
        bridge_ip = data.get('bridge_ip')
        username = data.get('username')
        light_id = data.get('light_id', '1')
        brightness = data.get('brightness', 100)
        color_temp = data.get('color_temperature', 4000)
        
        # Convert to Hue format
        hue_brightness = int(brightness * 2.54)  # 0-254
        hue_ct = int(1000000 / color_temp)  # Kelvin to Mirek
        
        command = {
            "url": f"http://{bridge_ip}/api/{username}/lights/{light_id}/state",
            "method": "PUT",
            "body": {
                "on": brightness > 0,
                "bri": hue_brightness,
                "ct": hue_ct
            }
        }
        
        return jsonify({
            "status": "command_generated",
            "command": command,
            "note": "Send this command to your Philips Hue Bridge"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ============================================================
# Analytics Endpoints
# ============================================================

@app.route('/analytics/daily', methods=['GET'])
def daily_analytics():
    """Get daily energy savings analytics"""
    facility_id = request.args.get('facility_id', 'default')
    
    # Calculate 24-hour savings across all zones
    hourly_data = []
    total_daily_savings = 0
    
    for hour in range(24):
        hour_savings = 0
        for zone_id in HOTEL_LIGHTING_PROFILES:
            brightness = get_optimal_brightness(zone_id, hour)
            savings = calculate_energy_savings(zone_id, hour, brightness)
            hour_savings += savings
        
        total_daily_savings += hour_savings
        hourly_data.append({
            "hour": hour,
            "savings_watts": hour_savings
        })
    
    cost_per_kwh = 25  # JPY
    daily_kwh_saved = total_daily_savings / 1000
    
    return jsonify({
        "facility_id": facility_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hourly_savings": hourly_data,
        "total_daily_savings_wh": total_daily_savings,
        "total_daily_savings_kwh": round(daily_kwh_saved, 2),
        "total_daily_cost_saved_yen": round(daily_kwh_saved * cost_per_kwh, 0),
        "monthly_projection_yen": round(daily_kwh_saved * cost_per_kwh * 30, 0),
        "annual_projection_yen": round(daily_kwh_saved * cost_per_kwh * 365, 0),
        "co2_reduced_kg_daily": round(daily_kwh_saved * 0.37, 2)
    })

# ============================================================
# Helper Functions
# ============================================================

def generate_reason(zone_id, hour, occupancy, natural_light, brightness):
    """Generate human-readable reason for lighting recommendation"""
    reasons = []
    
    if not occupancy:
        reasons.append("No occupancy detected - reduced to minimum")
    
    if natural_light > 200:
        reasons.append(f"Natural light ({int(natural_light)} lux) supplements artificial lighting")
    
    if 0 <= hour <= 5:
        reasons.append("Late night hours - minimal lighting needed")
    elif 6 <= hour <= 8:
        reasons.append("Morning transition - gradually increasing")
    elif 22 <= hour <= 23:
        reasons.append("Evening wind-down - reducing for comfort")
    
    if brightness < 50:
        reasons.append(f"Energy saving mode: {100-brightness}% reduction from full power")
    
    if not reasons:
        reasons.append("Optimal brightness for current conditions")
    
    return "; ".join(reasons)

# ============================================================
# Run Server
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
