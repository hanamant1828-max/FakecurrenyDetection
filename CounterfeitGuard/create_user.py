#!/usr/bin/env python
"""Script to create a user account"""
from app import app, db, User

def create_user(username, password):
    with app.app_context():
        # Check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            print(f"User '{username}' already exists!")
            return False
        
        # Create new user
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        print(f"User '{username}' created successfully!")
        return True

if __name__ == '__main__':
    # Create a demo user
    username = "admin"
    password = "admin123"
    
    create_user(username, password)
    print(f"\nLogin credentials:")
    print(f"Username: {username}")
    print(f"Password: {password}")
