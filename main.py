from CounterfeitGuard.app import app, db

# Initialize database tables on import
with app.app_context():
    db.create_all()
    print("Database initialized successfully")

if __name__ == "__main__":
    # Run the Flask app (only for local development)
    app.run(host="0.0.0.0", port=5000, debug=True)
