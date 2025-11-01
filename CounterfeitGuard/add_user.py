#!/usr/bin/env python
"""Simple script to create a user account without loading the model"""
import os
import sys
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///currency_detector.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

with app.app_context():
    # Create a demo user
    username = "admin"
    password = "admin123"
    
    # Check if user already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        print(f"User '{username}' already exists!")
    else:
        # Create new user
        user = User(username=username)
        user.password_hash = generate_password_hash(password)
        db.session.add(user)
        db.session.commit()
        print(f"User '{username}' created successfully!")
    
    print(f"\nLogin credentials:")
    print(f"Username: {username}")
    print(f"Password: {password}")
