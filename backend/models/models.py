from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    budget_preference = db.Column(db.String(20), default='medium')
    vibe_preference = db.Column(db.String(50), default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UserPreference(db.Model):
    __tablename__ = 'user_preferences'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    visit_count = db.Column(db.Integer, default=0)
    avg_rating = db.Column(db.Float, default=0.0)
    preferred_time = db.Column(db.String(20))
    last_visited = db.Column(db.DateTime)

class Place(db.Model):
    __tablename__ = 'places'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    subcategory = db.Column(db.String(100))
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    address = db.Column(db.String(500))
    rating = db.Column(db.Float, default=0.0)
    review_count = db.Column(db.Integer, default=0)
    price_level = db.Column(db.String(10))
    description = db.Column(db.Text)
    website = db.Column(db.String(500))
    phone = db.Column(db.String(50))
    hours = db.Column(db.JSON)
    images = db.Column(db.JSON)
    source = db.Column(db.String(50), default='api')
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    data = db.Column(db.JSON)
    ai_score = db.Column(db.Float, default=0.0)
    sentiment_score = db.Column(db.Float, default=0.0)
    crowd_level = db.Column(db.String(20), default='medium')

    def to_dict(self):
        return {
            'id': self.id, 'name': self.name, 'category': self.category,
            'subcategory': self.subcategory, 'latitude': self.latitude,
            'longitude': self.longitude, 'address': self.address,
            'rating': self.rating, 'review_count': self.review_count,
            'price_level': self.price_level, 'description': self.description,
            'website': self.website, 'phone': self.phone,
            'hours': self.hours, 'images': self.images,
            'source': self.source, 'ai_score': self.ai_score,
            'sentiment_score': self.sentiment_score,
            'crowd_level': self.crowd_level, 'data': self.data
        }

class Visit(db.Model):
    __tablename__ = 'visits'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    place_id = db.Column(db.Integer, db.ForeignKey('places.id'), nullable=False)
    category = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    duration_minutes = db.Column(db.Integer)
    user_rating = db.Column(db.Float)

class ScrapedData(db.Model):
    __tablename__ = 'scraped_data'
    id = db.Column(db.Integer, primary_key=True)
    place_id = db.Column(db.Integer, db.ForeignKey('places.id'), nullable=False)
    source = db.Column(db.String(50), nullable=False)
    raw_data = db.Column(db.JSON)
    sentiment = db.Column(db.Float)
    scraped_at = db.Column(db.DateTime, default=datetime.utcnow)
