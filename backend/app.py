import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///./location_guide.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)

users_db = {}  # Simple in-memory user storage

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def get_time_of_day():
    hour = datetime.now().hour
    if 6 <= hour < 12: return 'morning'
    elif 12 <= hour < 17: return 'afternoon'
    elif 17 <= hour < 21: return 'evening'
    else: return 'night'

def predict_crowd(category, hour):
    patterns = {
        'restaurant': {11:'low',12:'high',13:'medium',18:'high',19:'high',20:'medium',21:'low'},
        'cafe': {7:'medium',8:'high',9:'high',10:'medium',11:'low',14:'low',15:'medium',16:'medium',17:'low'},
        'mall': {10:'low',11:'medium',12:'high',13:'high',14:'medium',15:'medium',16:'high',17:'high',18:'medium',19:'low',20:'low'},
        'park': {6:'low',7:'medium',8:'high',9:'high',10:'medium',13:'low',15:'medium',16:'medium',17:'high',18:'high',19:'medium',20:'low'},
        'temple': {6:'high',7:'high',8:'medium',9:'medium',10:'low',11:'low',16:'medium',17:'medium',18:'high',19:'high'},
        'beach': {6:'low',7:'medium',8:'medium',9:'high',10:'high',11:'high',12:'high',13:'high',14:'medium',15:'medium',16:'high',17:'high',18:'medium'},
        'bar': {12:'low',17:'low',18:'low',19:'medium',20:'high',21:'high',22:'high',23:'high',0:'medium'},
        'museum': {9:'low',10:'medium',11:'medium',12:'high',13:'high',14:'medium',15:'medium',16:'low',17:'low'},
    }
    return patterns.get(category, {}).get(hour, 'medium')

def get_best_time(category):
    times = {
        'restaurant': 'Evening (6-9 PM)', 'cafe': 'Morning (8-11 AM)',
        'park': 'Morning (6-9 AM)', 'museum': 'Afternoon (2-5 PM)',
        'mall': 'Evening (6-9 PM)', 'bar': 'Night (8 PM - 12 AM)',
        'temple': 'Early Morning (6-8 AM)', 'beach': 'Late Afternoon (4-6 PM)',
        'gym': 'Morning (6-8 AM)', 'hospital': 'Any time (24/7)',
    }
    return times.get(category, 'Anytime')

def get_category_data(category):
    data = {
        'restaurant': {'names': ['Spice Garden', 'The Golden Fork', 'Ocean Breeze Seafood', 'Tandoori Nights', 'Pasta Paradise', 'Sushi Master', 'Burger Republic', 'Thai Orchid'], 'icon': '🍽️'},
        'cafe': {'names': ['Brew & Bean', 'The Cozy Corner', 'Artisan Coffee', 'Tea House', 'Pastry Dreams', 'Cafe Mocha'], 'icon': '☕'},
        'park': {'names': ['Sunset Park', 'Riverside Gardens', 'Central Green', 'Botanical Haven', 'Nature Trail Park', 'Lakeside Park'], 'icon': '🌳'},
        'mall': {'names': ['Grand Plaza Mall', 'City Center', 'Royal Shopping Hub', 'Metro Mall', 'The Grand Arcade', 'Skyline Mall'], 'icon': '🛍️'},
        'temple': {'names': ['Ancient Shiva Temple', 'Golden Pagoda', 'Sacred Heart Church', 'Marble Jain Temple', 'Sunrise Mosque', 'Lotus Meditation Center'], 'icon': '🛕'},
        'beach': {'names': ['Crystal Cove Beach', 'Palm Shore Beach', 'Sunset Beach', 'Golden Sands', 'Coral Bay', 'Tropical Paradise Beach'], 'icon': '🏖️'},
        'bar': {'names': ['The Rooftop Lounge', 'Craft & Draft', 'Midnight Blues', 'The Speakeasy', 'Sky Bar'], 'icon': '🍸'},
        'museum': {'names': ['National Art Museum', 'Science Discovery Center', 'History Museum', 'Modern Art Gallery', 'Cultural Heritage Museum'], 'icon': '🏛️'},
        'hospital': {'names': ['City Medical Center', 'Health First Hospital', 'Care Plus Clinic'], 'icon': '🏥'},
        'gym': {'names': ['FitZone Gym', 'Iron Paradise', 'PowerHouse Fitness', 'Zen Yoga Studio'], 'icon': '💪'},
    }
    return data.get(category, {'names': ['Place'], 'icon': '📍'})

def generate_places(lat, lng, category=None):
    categories = ['restaurant', 'cafe', 'park', 'mall', 'temple', 'beach', 'bar', 'museum', 'hospital', 'gym']
    if category and category in categories:
        categories = [category]
    
    places = []
    for cat in categories:
        cat_data = get_category_data(cat)
        for i, name in enumerate(cat_data['names'][:random.randint(3, 6)]):
            offset_lat = (random.random() - 0.5) * 0.04
            offset_lng = (random.random() - 0.5) * 0.04
            place_lat = lat + offset_lat
            place_lng = lng + offset_lng
            dist = calculate_distance(lat, lng, place_lat, place_lng)
            rating = round(random.uniform(3.5, 5.0), 1)
            hour = datetime.now().hour
            crowd = predict_crowd(cat, hour)
            sentiment = round(random.uniform(0.3, 0.9), 2)
            ai_score = round(min(100, rating * 15 + (10 if crowd == 'low' else 5 if crowd == 'medium' else -5) + sentiment * 10), 1)
            
            place_data = {
                'name': name,
                'category': cat,
                'subcategory': cat_data['icon'],
                'latitude': round(place_lat, 6),
                'longitude': round(place_lng, 6),
                'address': f'{random.randint(1,999)} {cat.title()} Street',
                'rating': rating,
                'review_count': random.randint(50, 2000),
                'price_level': random.choice(['$', '$$', '$$$', '$$$$']),
                'distance': round(dist, 2),
                'crowd_level': crowd,
                'best_time': get_best_time(cat),
                'sentiment_score': sentiment,
                'ai_score': ai_score,
                'description': f'Popular {cat} with excellent reviews',
                'phone': f'+91 {random.randint(9000000000, 9999999999)}',
                'hours': {'open': '9:00 AM', 'close': '10:00 PM'},
                'website': f'www.{name.lower().replace(" ", "")}.com',
                'data': {
                    'food_quality': round(random.uniform(3.5, 5.0), 1) if cat == 'restaurant' else None,
                    'service': round(random.uniform(3.5, 5.0), 1) if cat in ['restaurant', 'hotel'] else None,
                    'cleanliness': round(random.uniform(3.5, 5.0), 1),
                    'activities': ['Swimming', 'Surfing', 'Beach Volleyball'] if cat == 'beach' else None,
                    'ritual_timings': ['6:00 AM', '12:00 PM', '6:00 PM'] if cat == 'temple' else None,
                    'dress_code': 'Modest clothing required' if cat == 'temple' else None,
                    'parking': 'Available' if cat in ['mall', 'museum'] else 'Street parking',
                    'events': 'Weekend Sale 50% Off' if cat == 'mall' else None,
                }
            }
            places.append(place_data)
    
    places.sort(key=lambda x: x['distance'])
    return places

# HTML Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

@app.route('/places')
def places_page():
    return render_template('places.html')

# Auth Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Username, email, and password required'}), 400
    if data['email'] in users_db:
        return jsonify({'error': 'Email already exists'}), 409
    user_id = len(users_db) + 1
    users_db[data['email']] = {'id': user_id, 'username': data['username'], 'email': data['email']}
    return jsonify({'message': 'Registered', 'user_id': user_id}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400
    if data['email'] not in users_db:
        return jsonify({'error': 'Invalid credentials'}), 401
    user = users_db[data['email']]
    return jsonify({'message': 'Logged in', 'user_id': user['id']}), 200

# Location & AI Routes
@app.route('/location', methods=['POST'])
def receive_location():
    data = request.get_json()
    if not data or 'latitude' not in data or 'longitude' not in data:
        return jsonify({'error': 'Latitude and longitude required'}), 400
    lat, lng = data['latitude'], data['longitude']
    category = data.get('category')
    user_id = data.get('user_id')
    
    places = generate_places(lat, lng, category)
    
    ai_recs = sorted(places, key=lambda x: x.get('ai_score', 0), reverse=True)[:5]
    trending = sorted(places, key=lambda x: x.get('rating', 0), reverse=True)[:5]
    
    return jsonify({
        'user_location': {'latitude': lat, 'longitude': lng},
        'nearby_places': places,
        'ai_recommendations': ai_recs,
        'trending_places': trending,
        'time_of_day': get_time_of_day(),
        'count': len(places)
    }), 200

@app.route('/api/ai/recommendations', methods=['POST'])
def ai_recommendations():
    data = request.get_json() or {}
    lat = data.get('latitude', 20.5937)
    lng = data.get('longitude', 78.9629)
    places = generate_places(lat, lng)
    recs = sorted(places, key=lambda x: x.get('ai_score', 0), reverse=True)[:10]
    return jsonify({'recommendations': recs, 'count': len(recs)}), 200

@app.route('/api/places')
def get_places():
    lat = request.args.get('lat', 20.5937, type=float)
    lng = request.args.get('lng', 78.9629, type=float)
    places = generate_places(lat, lng)
    return jsonify({'places': places, 'count': len(places)}), 200

@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'}), 200
