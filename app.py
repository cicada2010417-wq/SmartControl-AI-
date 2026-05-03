"""
SmartControl AI - Production Lighting Control System v3.0
Real-time sensor-driven automatic lighting control with AI learning.

Features:
- Aqara sensor data ingestion (motion, light level)
- AI decision engine for optimal brightness
- Philips Hue integration for lighting control
- Custom zone management per facility
- Auto-learning pipeline
- Emergency override (full brightness)
- Minimum 20-30% brightness (never fully dark)
"""

import os
import json
import datetime
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ============================================================
# DATABASE SETUP
# ============================================================

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get('DATABASE_URL', '')

def get_db():
    """Get database connection"""
    conn = psycopg2.connect(DATABASE_URL, sslmode='require')
    return conn

def init_db():
    """Initialize database tables"""
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Facilities table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS facilities (
                id SERIAL PRIMARY KEY,
                facility_id VARCHAR(100) UNIQUE NOT NULL,
                facility_name VARCHAR(200) NOT NULL,
                facility_type VARCHAR(50) DEFAULT 'hotel',
                owner_id VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings JSONB DEFAULT '{}'
            )
        ''')
        
        # Zones table - customizable per facility
        cur.execute('''
            CREATE TABLE IF NOT EXISTS zones (
                id SERIAL PRIMARY KEY,
                zone_id VARCHAR(100) UNIQUE NOT NULL,
                facility_id VARCHAR(100) NOT NULL,
                zone_name VARCHAR(200) NOT NULL,
                zone_type VARCHAR(50) DEFAULT 'general',
                min_brightness INTEGER DEFAULT 25,
                max_brightness INTEGER DEFAULT 100,
                color_temp_min INTEGER DEFAULT 2700,
                color_temp_max INTEGER DEFAULT 6500,
                hue_light_ids JSONB DEFAULT '[]',
                aqara_sensor_ids JSONB DEFAULT '[]',
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                settings JSONB DEFAULT '{}'
            )
        ''')
        
        # Sensor readings table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id SERIAL PRIMARY KEY,
                zone_id VARCHAR(100) NOT NULL,
                facility_id VARCHAR(100) NOT NULL,
                sensor_type VARCHAR(50) NOT NULL,
                value FLOAT NOT NULL,
                raw_data JSONB DEFAULT '{}',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Lighting actions table (what AI decided)
        cur.execute('''
            CREATE TABLE IF NOT EXISTS lighting_actions (
                id SERIAL PRIMARY KEY,
                zone_id VARCHAR(100) NOT NULL,
                facility_id VARCHAR(100) NOT NULL,
                brightness INTEGER NOT NULL,
                color_temperature INTEGER,
                reason VARCHAR(500),
                sensor_data JSONB DEFAULT '{}',
                energy_saved_watts FLOAT DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # AI model metadata
        cur.execute('''
            CREATE TABLE IF NOT EXISTS ai_models (
                id SERIAL PRIMARY KEY,
                facility_id VARCHAR(100) NOT NULL,
                model_version INTEGER DEFAULT 1,
                accuracy FLOAT DEFAULT 0,
                training_samples INTEGER DEFAULT 0,
                model_data BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Emergency events
        cur.execute('''
            CREATE TABLE IF NOT EXISTS emergency_events (
                id SERIAL PRIMARY KEY,
                facility_id VARCHAR(100) NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        ''')
        
        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"DB init error: {e}")
        return False

# Initialize on startup
db_connected = False
try:
    db_connected = init_db()
except:
    pass

# ============================================================
# AI DECISION ENGINE
# ============================================================

class LightingAI:
    """AI engine that decides optimal brightness based on sensor data"""
    
    # Base rules (used before AI has enough learning data)
    RULES = {
        'high_occupancy_dark': {'brightness': 100, 'reason': 'High occupancy, low ambient light'},
        'high_occupancy_bright': {'brightness': 70, 'reason': 'High occupancy, sufficient ambient light'},
        'medium_occupancy_dark': {'brightness': 80, 'reason': 'Medium occupancy, low ambient light'},
        'medium_occupancy_bright': {'brightness': 50, 'reason': 'Medium occupancy, good ambient light'},
        'low_occupancy_dark': {'brightness': 60, 'reason': 'Low occupancy, low ambient light'},
        'low_occupancy_bright': {'brightness': 35, 'reason': 'Low occupancy, good ambient light'},
        'no_occupancy': {'brightness': 25, 'reason': 'No occupancy detected - minimum safe level'},
        'emergency': {'brightness': 100, 'reason': 'EMERGENCY - Full brightness activated'},
    }
    
    # Time-based adjustments
    TIME_ADJUSTMENTS = {
        'late_night': {'hours': (0, 5), 'factor': 0.8, 'min': 20},      # 0:00-5:00
        'early_morning': {'hours': (5, 7), 'factor': 0.9, 'min': 25},   # 5:00-7:00
        'morning': {'hours': (7, 10), 'factor': 1.0, 'min': 25},        # 7:00-10:00
        'midday': {'hours': (10, 14), 'factor': 0.85, 'min': 25},       # 10:00-14:00 (natural light)
        'afternoon': {'hours': (14, 18), 'factor': 0.9, 'min': 25},     # 14:00-18:00
        'evening': {'hours': (18, 22), 'factor': 1.0, 'min': 25},       # 18:00-22:00
        'night': {'hours': (22, 24), 'factor': 0.85, 'min': 20},        # 22:00-24:00
    }
    
    def __init__(self):
        self.learned_models = {}  # facility_id -> trained model
    
    def decide(self, zone_id, facility_id, motion_detected, people_count, 
               ambient_lux, hour, is_emergency=False, zone_settings=None):
        """
        Main AI decision function.
        
        Args:
            zone_id: Zone identifier
            facility_id: Facility identifier
            motion_detected: Boolean - is motion detected
            people_count: Integer - estimated number of people (0 = no one)
            ambient_lux: Float - ambient light level in lux (0-1000+)
            hour: Integer - current hour (0-23)
            is_emergency: Boolean - emergency mode
            zone_settings: Dict - zone-specific settings (min/max brightness)
            
        Returns:
            Dict with brightness, color_temperature, reason, energy_saved_watts
        """
        
        # Emergency override - always full brightness
        if is_emergency:
            return {
                'brightness': 100,
                'color_temperature': 5000,
                'reason': 'EMERGENCY - Full brightness activated',
                'energy_saved_watts': 0,
                'mode': 'emergency'
            }
        
        # Get zone settings
        min_brightness = 25  # Never go below this - "open for business" level
        max_brightness = 100
        if zone_settings:
            min_brightness = zone_settings.get('min_brightness', 25)
            max_brightness = zone_settings.get('max_brightness', 100)
        
        # Determine occupancy level
        if people_count == 0 and not motion_detected:
            occupancy = 'none'
        elif people_count <= 2 or (motion_detected and people_count == 0):
            occupancy = 'low'
        elif people_count <= 5:
            occupancy = 'medium'
        else:
            occupancy = 'high'
        
        # Determine ambient light level
        # < 100 lux = dark, 100-300 = dim, 300-500 = moderate, > 500 = bright
        if ambient_lux < 100:
            ambient = 'dark'
        elif ambient_lux < 300:
            ambient = 'dim'
        elif ambient_lux < 500:
            ambient = 'moderate'
        else:
            ambient = 'bright'
        
        # Calculate base brightness from rules
        brightness = self._calculate_base_brightness(occupancy, ambient)
        
        # Apply time-based adjustment
        time_factor, time_min = self._get_time_adjustment(hour)
        brightness = int(brightness * time_factor)
        min_brightness = max(min_brightness, time_min)
        
        # Apply ambient light compensation
        if ambient_lux > 300:
            # Reduce artificial light when natural light is available
            reduction = min(20, int((ambient_lux - 300) / 50))
            brightness -= reduction
        
        # Enforce min/max bounds
        brightness = max(min_brightness, min(max_brightness, brightness))
        
        # Calculate color temperature (warmer at night, cooler during day)
        color_temp = self._calculate_color_temp(hour, occupancy)
        
        # Calculate energy savings (compared to 100% always-on)
        energy_saved_watts = self._calculate_savings(brightness, zone_settings)
        
        # Generate reason
        reason = self._generate_reason(occupancy, ambient, hour, brightness)
        
        return {
            'brightness': brightness,
            'color_temperature': color_temp,
            'reason': reason,
            'energy_saved_watts': energy_saved_watts,
            'mode': 'ai_auto',
            'inputs': {
                'motion_detected': motion_detected,
                'people_count': people_count,
                'ambient_lux': ambient_lux,
                'hour': hour,
                'occupancy_level': occupancy,
                'ambient_level': ambient
            }
        }
    
    def _calculate_base_brightness(self, occupancy, ambient):
        """Calculate base brightness from occupancy and ambient light"""
        matrix = {
            ('high', 'dark'): 100,
            ('high', 'dim'): 90,
            ('high', 'moderate'): 75,
            ('high', 'bright'): 60,
            ('medium', 'dark'): 85,
            ('medium', 'dim'): 75,
            ('medium', 'moderate'): 55,
            ('medium', 'bright'): 40,
            ('low', 'dark'): 65,
            ('low', 'dim'): 55,
            ('low', 'moderate'): 40,
            ('low', 'bright'): 30,
            ('none', 'dark'): 30,
            ('none', 'dim'): 25,
            ('none', 'moderate'): 25,
            ('none', 'bright'): 20,
        }
        return matrix.get((occupancy, ambient), 50)
    
    def _get_time_adjustment(self, hour):
        """Get time-based brightness adjustment factor"""
        for period, config in self.TIME_ADJUSTMENTS.items():
            start, end = config['hours']
            if start <= hour < end:
                return config['factor'], config['min']
        return 1.0, 25
    
    def _calculate_color_temp(self, hour, occupancy):
        """Calculate optimal color temperature"""
        # Warmer (2700K) at night for comfort, cooler (5000K) during day for alertness
        if hour < 6 or hour > 21:
            return 2700  # Warm white - relaxing
        elif hour < 9:
            return 3500  # Transitional morning
        elif hour < 17:
            return 4500  # Cool white - productive
        elif hour < 21:
            return 3500  # Transitional evening
        else:
            return 2700  # Warm white
    
    def _calculate_savings(self, brightness, zone_settings):
        """Calculate energy savings in watts"""
        # Assume average zone has 200W of lighting at 100%
        max_watts = 200
        if zone_settings and 'max_watts' in zone_settings:
            max_watts = zone_settings['max_watts']
        
        current_watts = max_watts * (brightness / 100.0)
        saved_watts = max_watts - current_watts
        return round(saved_watts, 1)
    
    def _generate_reason(self, occupancy, ambient, hour, brightness):
        """Generate human-readable reason for the decision"""
        reasons = []
        
        if occupancy == 'none':
            reasons.append("No people detected")
        elif occupancy == 'low':
            reasons.append("Low occupancy")
        elif occupancy == 'medium':
            reasons.append("Medium occupancy")
        else:
            reasons.append("High occupancy")
        
        if ambient in ('bright', 'moderate'):
            reasons.append("good natural light available")
        elif ambient == 'dim':
            reasons.append("low ambient light")
        else:
            reasons.append("dark environment")
        
        if hour < 6 or hour > 22:
            reasons.append("late night hours")
        
        return f"{' + '.join(reasons)} → {brightness}% brightness"


# Global AI instance
lighting_ai = LightingAI()

# ============================================================
# API ENDPOINTS
# ============================================================

# --- Health Check ---
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'version': 'SmartControl AI v3.0',
        'database': 'connected' if db_connected else 'not connected',
        'features': [
            'sensor_ingestion',
            'ai_decision_engine', 
            'philips_hue_control',
            'zone_management',
            'auto_learning',
            'emergency_override'
        ]
    })

# --- Facility Management ---
@app.route('/facility/register', methods=['POST'])
def register_facility():
    """Register a new facility"""
    data = request.json
    facility_id = data.get('facility_id')
    facility_name = data.get('facility_name')
    facility_type = data.get('facility_type', 'hotel')
    owner_id = data.get('owner_id', '')
    
    if not facility_id or not facility_name:
        return jsonify({'error': 'facility_id and facility_name required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO facilities (facility_id, facility_name, facility_type, owner_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (facility_id) DO UPDATE SET 
                facility_name = EXCLUDED.facility_name,
                facility_type = EXCLUDED.facility_type
            RETURNING id
        ''', (facility_id, facility_name, facility_type, owner_id))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'status': 'registered',
            'facility_id': facility_id,
            'facility_name': facility_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/facility/<facility_id>', methods=['GET'])
def get_facility(facility_id):
    """Get facility info"""
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM facilities WHERE facility_id = %s', (facility_id,))
        facility = cur.fetchone()
        
        cur.execute('SELECT * FROM zones WHERE facility_id = %s AND is_active = TRUE', (facility_id,))
        zones = cur.fetchall()
        
        cur.close()
        conn.close()
        
        if not facility:
            return jsonify({'error': 'Facility not found'}), 404
        
        return jsonify({
            'facility': dict(facility),
            'zones': [dict(z) for z in zones]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Zone Management ---
@app.route('/zones/create', methods=['POST'])
def create_zone():
    """Create a new zone for a facility"""
    data = request.json
    zone_id = data.get('zone_id')
    facility_id = data.get('facility_id')
    zone_name = data.get('zone_name')
    zone_type = data.get('zone_type', 'general')
    min_brightness = data.get('min_brightness', 25)
    max_brightness = data.get('max_brightness', 100)
    hue_light_ids = data.get('hue_light_ids', [])
    aqara_sensor_ids = data.get('aqara_sensor_ids', [])
    settings = data.get('settings', {})
    
    if not all([zone_id, facility_id, zone_name]):
        return jsonify({'error': 'zone_id, facility_id, and zone_name required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO zones (zone_id, facility_id, zone_name, zone_type, 
                             min_brightness, max_brightness, hue_light_ids, 
                             aqara_sensor_ids, settings)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (zone_id) DO UPDATE SET
                zone_name = EXCLUDED.zone_name,
                zone_type = EXCLUDED.zone_type,
                min_brightness = EXCLUDED.min_brightness,
                max_brightness = EXCLUDED.max_brightness,
                hue_light_ids = EXCLUDED.hue_light_ids,
                aqara_sensor_ids = EXCLUDED.aqara_sensor_ids,
                settings = EXCLUDED.settings
            RETURNING id
        ''', (zone_id, facility_id, zone_name, zone_type, min_brightness, 
              max_brightness, json.dumps(hue_light_ids), 
              json.dumps(aqara_sensor_ids), json.dumps(settings)))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'status': 'created',
            'zone_id': zone_id,
            'zone_name': zone_name,
            'facility_id': facility_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/zones/<facility_id>', methods=['GET'])
def list_zones(facility_id):
    """List all zones for a facility"""
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT * FROM zones WHERE facility_id = %s AND is_active = TRUE
            ORDER BY created_at
        ''', (facility_id,))
        zones = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify({
            'facility_id': facility_id,
            'zones': [dict(z) for z in zones],
            'count': len(zones)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/zones/update', methods=['PUT'])
def update_zone():
    """Update zone settings"""
    data = request.json
    zone_id = data.get('zone_id')
    
    if not zone_id:
        return jsonify({'error': 'zone_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor()
        
        updates = []
        values = []
        
        if 'zone_name' in data:
            updates.append('zone_name = %s')
            values.append(data['zone_name'])
        if 'min_brightness' in data:
            updates.append('min_brightness = %s')
            values.append(data['min_brightness'])
        if 'max_brightness' in data:
            updates.append('max_brightness = %s')
            values.append(data['max_brightness'])
        if 'hue_light_ids' in data:
            updates.append('hue_light_ids = %s')
            values.append(json.dumps(data['hue_light_ids']))
        if 'aqara_sensor_ids' in data:
            updates.append('aqara_sensor_ids = %s')
            values.append(json.dumps(data['aqara_sensor_ids']))
        if 'settings' in data:
            updates.append('settings = %s')
            values.append(json.dumps(data['settings']))
        if 'is_active' in data:
            updates.append('is_active = %s')
            values.append(data['is_active'])
        
        if updates:
            values.append(zone_id)
            cur.execute(f'''
                UPDATE zones SET {', '.join(updates)} WHERE zone_id = %s
            ''', values)
            conn.commit()
        
        cur.close()
        conn.close()
        
        return jsonify({'status': 'updated', 'zone_id': zone_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/zones/delete', methods=['DELETE'])
def delete_zone():
    """Soft-delete a zone"""
    data = request.json
    zone_id = data.get('zone_id')
    
    if not zone_id:
        return jsonify({'error': 'zone_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('UPDATE zones SET is_active = FALSE WHERE zone_id = %s', (zone_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({'status': 'deleted', 'zone_id': zone_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Sensor Data Ingestion ---
@app.route('/sensor/push', methods=['POST'])
def push_sensor_data():
    """
    Receive sensor data from Aqara Hub webhook.
    This is the main entry point for real-time sensor data.
    
    Expected payload:
    {
        "zone_id": "lobby_1f",
        "facility_id": "hotel_sunrise",
        "sensors": {
            "motion": true/false,
            "people_count": 3,
            "ambient_lux": 250.5,
            "temperature": 24.5
        }
    }
    """
    data = request.json
    zone_id = data.get('zone_id')
    facility_id = data.get('facility_id')
    sensors = data.get('sensors', {})
    
    if not zone_id or not facility_id:
        return jsonify({'error': 'zone_id and facility_id required'}), 400
    
    # Extract sensor values
    motion_detected = sensors.get('motion', False)
    people_count = sensors.get('people_count', 0)
    ambient_lux = sensors.get('ambient_lux', 200)
    
    # Get current hour
    now = datetime.datetime.now()
    hour = now.hour
    
    # Get zone settings
    zone_settings = None
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM zones WHERE zone_id = %s', (zone_id,))
        zone = cur.fetchone()
        if zone:
            zone_settings = {
                'min_brightness': zone['min_brightness'],
                'max_brightness': zone['max_brightness'],
                'max_watts': zone.get('settings', {}).get('max_watts', 200) if zone.get('settings') else 200
            }
        cur.close()
        conn.close()
    except:
        pass
    
    # Check for emergency
    is_emergency = data.get('emergency', False)
    
    # AI Decision
    decision = lighting_ai.decide(
        zone_id=zone_id,
        facility_id=facility_id,
        motion_detected=motion_detected,
        people_count=people_count,
        ambient_lux=ambient_lux,
        hour=hour,
        is_emergency=is_emergency,
        zone_settings=zone_settings
    )
    
    # Store sensor reading and action in database
    try:
        conn = get_db()
        cur = conn.cursor()
        
        # Store sensor readings
        for sensor_type, value in sensors.items():
            if isinstance(value, (int, float)):
                cur.execute('''
                    INSERT INTO sensor_readings (zone_id, facility_id, sensor_type, value, raw_data)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (zone_id, facility_id, sensor_type, float(value), json.dumps(sensors)))
            elif isinstance(value, bool):
                cur.execute('''
                    INSERT INTO sensor_readings (zone_id, facility_id, sensor_type, value, raw_data)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (zone_id, facility_id, sensor_type, 1.0 if value else 0.0, json.dumps(sensors)))
        
        # Store lighting action
        cur.execute('''
            INSERT INTO lighting_actions (zone_id, facility_id, brightness, color_temperature, 
                                        reason, sensor_data, energy_saved_watts)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (zone_id, facility_id, decision['brightness'], decision['color_temperature'],
              decision['reason'], json.dumps(decision['inputs']), decision['energy_saved_watts']))
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB store error: {e}")
    
    # Generate Philips Hue commands
    hue_commands = generate_hue_commands(zone_id, decision)
    
    return jsonify({
        'status': 'ok',
        'zone_id': zone_id,
        'decision': decision,
        'hue_commands': hue_commands,
        'timestamp': now.isoformat()
    })

# --- Batch Sensor Update (for polling mode) ---
@app.route('/sensor/batch', methods=['POST'])
def batch_sensor_update():
    """
    Process multiple zone sensor updates at once.
    Used when polling all sensors periodically.
    
    Expected payload:
    {
        "facility_id": "hotel_sunrise",
        "zones": [
            {"zone_id": "lobby", "sensors": {"motion": true, "people_count": 5, "ambient_lux": 300}},
            {"zone_id": "corridor_1f", "sensors": {"motion": false, "people_count": 0, "ambient_lux": 50}}
        ]
    }
    """
    data = request.json
    facility_id = data.get('facility_id')
    zones_data = data.get('zones', [])
    
    if not facility_id or not zones_data:
        return jsonify({'error': 'facility_id and zones required'}), 400
    
    results = []
    now = datetime.datetime.now()
    hour = now.hour
    
    for zone_data in zones_data:
        zone_id = zone_data.get('zone_id')
        sensors = zone_data.get('sensors', {})
        
        motion_detected = sensors.get('motion', False)
        people_count = sensors.get('people_count', 0)
        ambient_lux = sensors.get('ambient_lux', 200)
        is_emergency = zone_data.get('emergency', False)
        
        # Get zone settings from DB
        zone_settings = None
        try:
            conn = get_db()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute('SELECT * FROM zones WHERE zone_id = %s', (zone_id,))
            zone = cur.fetchone()
            if zone:
                zone_settings = {
                    'min_brightness': zone['min_brightness'],
                    'max_brightness': zone['max_brightness'],
                    'max_watts': zone.get('settings', {}).get('max_watts', 200) if zone.get('settings') else 200
                }
            cur.close()
            conn.close()
        except:
            pass
        
        decision = lighting_ai.decide(
            zone_id=zone_id,
            facility_id=facility_id,
            motion_detected=motion_detected,
            people_count=people_count,
            ambient_lux=ambient_lux,
            hour=hour,
            is_emergency=is_emergency,
            zone_settings=zone_settings
        )
        
        hue_commands = generate_hue_commands(zone_id, decision)
        
        results.append({
            'zone_id': zone_id,
            'decision': decision,
            'hue_commands': hue_commands
        })
    
    return jsonify({
        'status': 'ok',
        'facility_id': facility_id,
        'results': results,
        'timestamp': now.isoformat()
    })

# --- Emergency Control ---
@app.route('/emergency/activate', methods=['POST'])
def activate_emergency():
    """Activate emergency mode - all lights to 100%"""
    data = request.json
    facility_id = data.get('facility_id')
    event_type = data.get('event_type', 'fire_alarm')
    
    if not facility_id:
        return jsonify({'error': 'facility_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Record emergency event
        cur.execute('''
            INSERT INTO emergency_events (facility_id, event_type)
            VALUES (%s, %s)
        ''', (facility_id, event_type))
        
        # Get all zones for this facility
        cur.execute('SELECT * FROM zones WHERE facility_id = %s AND is_active = TRUE', (facility_id,))
        zones = cur.fetchall()
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Generate full-brightness commands for all zones
        emergency_commands = []
        for zone in zones:
            decision = lighting_ai.decide(
                zone_id=zone['zone_id'],
                facility_id=facility_id,
                motion_detected=True,
                people_count=99,
                ambient_lux=0,
                hour=12,
                is_emergency=True,
                zone_settings=None
            )
            hue_commands = generate_hue_commands(zone['zone_id'], decision)
            emergency_commands.append({
                'zone_id': zone['zone_id'],
                'zone_name': zone['zone_name'],
                'brightness': 100,
                'hue_commands': hue_commands
            })
        
        return jsonify({
            'status': 'emergency_activated',
            'facility_id': facility_id,
            'event_type': event_type,
            'all_zones_full_brightness': True,
            'commands': emergency_commands
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/emergency/deactivate', methods=['POST'])
def deactivate_emergency():
    """Deactivate emergency mode - return to AI control"""
    data = request.json
    facility_id = data.get('facility_id')
    
    if not facility_id:
        return jsonify({'error': 'facility_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            UPDATE emergency_events SET is_active = FALSE, resolved_at = CURRENT_TIMESTAMP
            WHERE facility_id = %s AND is_active = TRUE
        ''', (facility_id,))
        conn.commit()
        cur.close()
        conn.close()
        
        return jsonify({
            'status': 'emergency_deactivated',
            'facility_id': facility_id,
            'message': 'Returning to AI auto-control mode'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Philips Hue Integration ---
def generate_hue_commands(zone_id, decision):
    """Generate Philips Hue API commands based on AI decision"""
    brightness = decision['brightness']
    color_temp = decision['color_temperature']
    
    # Convert brightness 0-100 to Hue 1-254
    hue_brightness = max(1, int(brightness * 254 / 100))
    
    # Convert color temp K to Hue mirek (1000000/K)
    hue_mirek = int(1000000 / color_temp) if color_temp > 0 else 250
    hue_mirek = max(153, min(500, hue_mirek))  # Hue range: 153-500
    
    return {
        'state': {
            'on': brightness > 0,
            'bri': hue_brightness,
            'ct': hue_mirek,
            'transitiontime': 10  # 1 second smooth transition
        },
        'brightness_percent': brightness,
        'color_temp_kelvin': color_temp,
        'note': 'Send this to PUT /api/<username>/lights/<light_id>/state'
    }

@app.route('/hue/setup', methods=['GET'])
def hue_setup_guide():
    """Guide for setting up Philips Hue Bridge connection"""
    return jsonify({
        'setup_steps': [
            '1. Connect Philips Hue Bridge to your network via Ethernet',
            '2. Find Bridge IP: visit https://discovery.meethue.com/',
            '3. Press the link button on the Bridge',
            '4. Within 30 seconds, call POST /hue/register with bridge_ip',
            '5. Save the returned username for future API calls',
            '6. Add light IDs to your zones via /zones/update'
        ],
        'required_hardware': {
            'bridge': 'Philips Hue Bridge (1 per facility)',
            'lights': 'Philips Hue White Ambiance bulbs (per zone)',
            'optional': 'Philips Hue motion sensor (alternative to Aqara)'
        },
        'api_base': 'http://<bridge_ip>/api/<username>',
        'endpoints': {
            'list_lights': 'GET /api/<username>/lights',
            'control_light': 'PUT /api/<username>/lights/<id>/state',
            'list_groups': 'GET /api/<username>/groups',
            'control_group': 'PUT /api/<username>/groups/<id>/action'
        }
    })

@app.route('/hue/register', methods=['POST'])
def hue_register():
    """Register with Philips Hue Bridge to get API username"""
    data = request.json
    bridge_ip = data.get('bridge_ip')
    
    if not bridge_ip:
        return jsonify({'error': 'bridge_ip required'}), 400
    
    return jsonify({
        'instruction': 'Press the link button on your Hue Bridge, then call this endpoint again within 30 seconds',
        'curl_command': f'curl -X POST http://{bridge_ip}/api -d \'{{"devicetype":"smartcontrol_ai#device"}}\'',
        'note': 'The response will contain your username. Save it and add to facility settings.'
    })

@app.route('/hue/command', methods=['POST'])
def hue_command():
    """
    Generate ready-to-send Hue API command.
    
    Payload:
    {
        "bridge_ip": "192.168.1.100",
        "username": "your-hue-username",
        "light_id": "1",
        "brightness": 75,
        "color_temperature": 3500
    }
    """
    data = request.json
    bridge_ip = data.get('bridge_ip')
    username = data.get('username')
    light_id = data.get('light_id')
    brightness = data.get('brightness', 100)
    color_temp = data.get('color_temperature', 4000)
    
    if not all([bridge_ip, username, light_id]):
        return jsonify({'error': 'bridge_ip, username, and light_id required'}), 400
    
    hue_bri = max(1, int(brightness * 254 / 100))
    hue_ct = int(1000000 / color_temp) if color_temp > 0 else 250
    hue_ct = max(153, min(500, hue_ct))
    
    api_url = f'http://{bridge_ip}/api/{username}/lights/{light_id}/state'
    body = {
        'on': brightness > 0,
        'bri': hue_bri,
        'ct': hue_ct,
        'transitiontime': 10
    }
    
    return jsonify({
        'url': api_url,
        'method': 'PUT',
        'body': body,
        'curl': f"curl -X PUT {api_url} -d '{json.dumps(body)}'"
    })

# --- Aqara Integration ---
@app.route('/aqara/webhook', methods=['POST'])
def aqara_webhook():
    """
    Receive webhook from Aqara Hub when sensor state changes.
    This is called automatically by Aqara when motion is detected/cleared.
    
    Aqara sends data in this format (simplified):
    {
        "msgType": "report",
        "deviceModel": "lumi.motion.agl04",
        "resourceId": "4.1.85",
        "value": "1",  // 1=motion detected, 0=no motion
        "deviceId": "xxx",
        "subjectId": "xxx"
    }
    """
    data = request.json
    
    # Map Aqara device to our zone (lookup in DB)
    device_id = data.get('deviceId', data.get('device_id', ''))
    msg_type = data.get('msgType', data.get('msg_type', ''))
    value = data.get('value', '0')
    resource_id = data.get('resourceId', data.get('resource_id', ''))
    
    # Determine sensor type from resource ID
    # Aqara motion sensor: resourceId 4.1.85 = motion
    # Aqara light sensor: resourceId 0.3.85 = illuminance
    sensor_type = 'motion'
    sensor_value = 0
    
    if '4.1.85' in str(resource_id):
        sensor_type = 'motion'
        sensor_value = 1 if str(value) == '1' else 0
    elif '0.3.85' in str(resource_id):
        sensor_type = 'ambient_lux'
        sensor_value = float(value) if value else 0
    elif '0.1.85' in str(resource_id):
        sensor_type = 'people_count'
        sensor_value = int(value) if value else 0
    
    # Find zone for this device
    zone_id = None
    facility_id = None
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('''
            SELECT zone_id, facility_id FROM zones 
            WHERE aqara_sensor_ids::text LIKE %s AND is_active = TRUE
            LIMIT 1
        ''', (f'%{device_id}%',))
        result = cur.fetchone()
        if result:
            zone_id = result['zone_id']
            facility_id = result['facility_id']
        cur.close()
        conn.close()
    except:
        pass
    
    if not zone_id:
        return jsonify({
            'status': 'received',
            'warning': 'Device not mapped to any zone',
            'device_id': device_id
        })
    
    # Get latest sensor data for this zone to make a complete decision
    sensors = {'motion': sensor_type == 'motion' and sensor_value == 1}
    if sensor_type == 'ambient_lux':
        sensors['ambient_lux'] = sensor_value
    if sensor_type == 'people_count':
        sensors['people_count'] = int(sensor_value)
    
    # Forward to main sensor processing
    internal_data = {
        'zone_id': zone_id,
        'facility_id': facility_id,
        'sensors': sensors
    }
    
    # Process through AI
    with app.test_request_context(json=internal_data):
        # Reuse push_sensor_data logic
        pass
    
    now = datetime.datetime.now()
    decision = lighting_ai.decide(
        zone_id=zone_id,
        facility_id=facility_id,
        motion_detected=sensors.get('motion', False),
        people_count=sensors.get('people_count', 0),
        ambient_lux=sensors.get('ambient_lux', 200),
        hour=now.hour,
        is_emergency=False,
        zone_settings=None
    )
    
    hue_commands = generate_hue_commands(zone_id, decision)
    
    return jsonify({
        'status': 'processed',
        'zone_id': zone_id,
        'facility_id': facility_id,
        'sensor_type': sensor_type,
        'sensor_value': sensor_value,
        'decision': decision,
        'hue_commands': hue_commands
    })

@app.route('/aqara/setup', methods=['GET'])
def aqara_setup_guide():
    """Guide for setting up Aqara sensors"""
    return jsonify({
        'setup_steps': [
            '1. Purchase Aqara Hub M2 and sensors (Motion P1, Light Sensor T1)',
            '2. Install Aqara Home app and add Hub to your Wi-Fi',
            '3. Pair sensors with Hub via app',
            '4. In Aqara Developer Portal (https://developer.aqara.com/), create an app',
            '5. Enable webhook notifications for your devices',
            '6. Set webhook URL to: https://smartcontrol-ai.onrender.com/aqara/webhook',
            '7. Map device IDs to zones via /zones/update (add to aqara_sensor_ids)',
            '8. Sensors will now automatically push data to this server'
        ],
        'recommended_sensors': {
            'motion': 'Aqara Motion Sensor P1 (lumi.motion.agl04) - ~3000 yen',
            'light': 'Aqara Light Sensor T1 (lumi.sen_ill.agl01) - ~2500 yen',
            'presence': 'Aqara Presence Sensor FP2 (lumi.motion.agl001) - ~12000 yen (people counting)',
            'hub': 'Aqara Hub M2 (lumi.gateway.aeu01) - ~5000 yen'
        },
        'webhook_url': 'https://smartcontrol-ai.onrender.com/aqara/webhook',
        'note': 'The FP2 sensor can count people and detect zones - ideal for larger areas'
    })

# --- Auto Learning ---
@app.route('/learn/trigger', methods=['POST'])
def trigger_learning():
    """
    Trigger AI model retraining based on collected data.
    The AI learns from past sensor readings and the resulting energy savings.
    """
    data = request.json
    facility_id = data.get('facility_id')
    
    if not facility_id:
        return jsonify({'error': 'facility_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get training data: sensor readings + lighting actions
        cur.execute('''
            SELECT la.brightness, la.sensor_data, la.energy_saved_watts,
                   la.zone_id, la.timestamp
            FROM lighting_actions la
            WHERE la.facility_id = %s
            ORDER BY la.timestamp DESC
            LIMIT 10000
        ''', (facility_id,))
        
        actions = cur.fetchall()
        
        if len(actions) < 50:
            cur.close()
            conn.close()
            return jsonify({
                'status': 'insufficient_data',
                'message': f'Need at least 50 data points, currently have {len(actions)}',
                'current_count': len(actions),
                'required': 50
            })
        
        # Prepare training data
        X = []
        y = []
        
        for action in actions:
            sensor_data = action['sensor_data'] if isinstance(action['sensor_data'], dict) else {}
            features = [
                1 if sensor_data.get('motion_detected', False) else 0,
                sensor_data.get('people_count', 0),
                sensor_data.get('ambient_lux', 200),
                sensor_data.get('hour', 12),
            ]
            X.append(features)
            y.append(action['brightness'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train model
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        accuracy = float(np.mean(scores))
        
        # Train final model
        model.fit(X, y)
        
        # Save model
        model_bytes = pickle.dumps(model)
        
        cur.execute('''
            INSERT INTO ai_models (facility_id, model_version, accuracy, training_samples, model_data)
            VALUES (%s, 
                    (SELECT COALESCE(MAX(model_version), 0) + 1 FROM ai_models WHERE facility_id = %s),
                    %s, %s, %s)
        ''', (facility_id, facility_id, accuracy, len(actions), psycopg2.Binary(model_bytes)))
        
        conn.commit()
        cur.close()
        conn.close()
        
        # Load into memory
        lighting_ai.learned_models[facility_id] = model
        
        return jsonify({
            'status': 'training_complete',
            'facility_id': facility_id,
            'accuracy_r2': round(accuracy, 4),
            'training_samples': len(actions),
            'message': f'Model trained with R²={accuracy:.4f} on {len(actions)} samples'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/learn/status', methods=['GET'])
def learning_status():
    """Check learning status for a facility"""
    facility_id = request.args.get('facility_id')
    
    if not facility_id:
        return jsonify({'error': 'facility_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get latest model info
        cur.execute('''
            SELECT model_version, accuracy, training_samples, created_at
            FROM ai_models WHERE facility_id = %s
            ORDER BY model_version DESC LIMIT 1
        ''', (facility_id,))
        model_info = cur.fetchone()
        
        # Get data count
        cur.execute('''
            SELECT COUNT(*) as count FROM lighting_actions WHERE facility_id = %s
        ''', (facility_id,))
        data_count = cur.fetchone()['count']
        
        cur.close()
        conn.close()
        
        return jsonify({
            'facility_id': facility_id,
            'data_points_collected': data_count,
            'min_required': 50,
            'ready_to_train': data_count >= 50,
            'current_model': dict(model_info) if model_info else None,
            'mode': 'ai_learned' if model_info else 'rule_based'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Analytics ---
@app.route('/analytics/summary', methods=['GET'])
def analytics_summary():
    """Get energy savings summary for a facility"""
    facility_id = request.args.get('facility_id')
    days = int(request.args.get('days', 7))
    
    if not facility_id:
        return jsonify({'error': 'facility_id required'}), 400
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute('''
            SELECT 
                DATE(timestamp) as date,
                AVG(brightness) as avg_brightness,
                SUM(energy_saved_watts) as total_saved_watts,
                COUNT(*) as decisions_made
            FROM lighting_actions 
            WHERE facility_id = %s AND timestamp > NOW() - INTERVAL '%s days'
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        ''', (facility_id, days))
        
        daily_stats = cur.fetchall()
        
        # Total savings
        cur.execute('''
            SELECT 
                SUM(energy_saved_watts) as total_saved,
                AVG(brightness) as avg_brightness,
                COUNT(*) as total_decisions
            FROM lighting_actions WHERE facility_id = %s
        ''', (facility_id,))
        totals = cur.fetchone()
        
        cur.close()
        conn.close()
        
        # Calculate cost savings (25 yen per kWh average in Japan)
        total_saved_kwh = (totals['total_saved'] or 0) / 1000 if totals['total_saved'] else 0
        cost_saved_yen = total_saved_kwh * 25
        co2_saved_kg = total_saved_kwh * 0.37  # Japan CO2 factor
        
        return jsonify({
            'facility_id': facility_id,
            'period_days': days,
            'daily_stats': [dict(d) for d in daily_stats],
            'totals': {
                'total_energy_saved_wh': totals['total_saved'] or 0,
                'average_brightness': round(totals['avg_brightness'] or 0, 1),
                'total_decisions': totals['total_decisions'] or 0,
                'estimated_cost_saved_yen': round(cost_saved_yen, 0),
                'co2_saved_kg': round(co2_saved_kg, 2)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Lighting Status (for iOS app) ---
@app.route('/lighting/status', methods=['GET'])
def lighting_status():
    """Get current lighting status for all zones in a facility"""
    facility_id = request.args.get('facility_id')
    
    if not facility_id:
        return jsonify({'error': 'facility_id required'}), 400
    
    now = datetime.datetime.now()
    
    try:
        conn = get_db()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute('''
            SELECT * FROM zones WHERE facility_id = %s AND is_active = TRUE
        ''', (facility_id,))
        zones = cur.fetchall()
        
        # Get latest action for each zone
        zone_statuses = []
        total_saved = 0
        
        for zone in zones:
            cur.execute('''
                SELECT brightness, color_temperature, energy_saved_watts, reason
                FROM lighting_actions 
                WHERE zone_id = %s
                ORDER BY timestamp DESC LIMIT 1
            ''', (zone['zone_id'],))
            latest = cur.fetchone()
            
            if latest:
                brightness = latest['brightness']
                color_temp = latest['color_temperature']
                saved = latest['energy_saved_watts']
                reason = latest['reason']
            else:
                # No data yet - use AI default
                decision = lighting_ai.decide(
                    zone_id=zone['zone_id'],
                    facility_id=facility_id,
                    motion_detected=False,
                    people_count=0,
                    ambient_lux=200,
                    hour=now.hour,
                    zone_settings={
                        'min_brightness': zone['min_brightness'],
                        'max_brightness': zone['max_brightness']
                    }
                )
                brightness = decision['brightness']
                color_temp = decision['color_temperature']
                saved = decision['energy_saved_watts']
                reason = decision['reason']
            
            total_saved += saved
            zone_statuses.append({
                'zone_id': zone['zone_id'],
                'zone_name': zone['zone_name'],
                'brightness': brightness,
                'color_temperature': color_temp,
                'target_lux': 300,
                'energy_saved_watts': saved,
                'is_ai_controlled': True,
                'reason': reason
            })
        
        cur.close()
        conn.close()
        
        return jsonify({
            'facility_id': facility_id,
            'timestamp': now.isoformat(),
            'current_hour': now.hour,
            'zones': zone_statuses,
            'total_energy_saved_watts': total_saved,
            'ai_mode': 'active'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
