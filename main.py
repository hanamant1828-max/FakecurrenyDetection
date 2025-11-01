
from CounterfeitGuard.app import app, db

if __name__ == "__main__":
    # Initialize database tables
    with app.app_context():
        db.create_all()
        print("Database initialized successfully")
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)
