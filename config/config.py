import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ai-location-guide-secret-key-2024')
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-key-2024')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///../location_guide.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
    GOOGLE_PLACES_API_KEY = os.environ.get('GOOGLE_PLACES_API_KEY', '')
    YELP_API_KEY = os.environ.get('YELP_API_KEY', '')
    TRIPADVISOR_API_KEY = os.environ.get('TRIPADVISOR_API_KEY', '')
    TWITTER_BEARER_TOKEN = os.environ.get('TWITTER_BEARER_TOKEN', '')
    GEOFENCE_RADIUS = float(os.environ.get('GEOFENCE_RADIUS', 5000))
    CACHE_TTL = int(os.environ.get('CACHE_TTL', 3600))
