from flask import Flask
from . import routes

def create_app():
    app = Flask(__name__)
    
    # Configure app settings
    app.config.from_mapping(
        SECRET_KEY='dev',  # Change this to a secure key in production
    )

    # Register blueprints and routes here (can be expanded later)
    app.register_blueprint(routes.bp)

    return app