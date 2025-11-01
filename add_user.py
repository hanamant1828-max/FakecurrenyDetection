#!/usr/bin/env python
"""Script to create a user account for the Currency Detector app"""
import sys
sys.path.insert(0, '.')

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
import os

# Create minimal Flask app just for database operations
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///CounterfeitGuard/instance/currency_detector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

def create_user(username, password):
    with app.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f"User '{username}' already exists!")
            return False
        
        # Create new user
        user = User(username=username)
        user.password_hash = generate_password_hash(password)
        db.session.add(user)
        db.session.commit()
        print(f"✓ User '{username}' created successfully!")
        return True

if __name__ == '__main__':
    # Create admin user
    username = "admin"
    password = "admin123"
    
    create_user(username, password)
    print(f"\nYour login credentials:")
    print(f"  Username: {username}")
    print(f"  Password: {password}")
    print("\nYou can now login to the Currency Detector app!")
