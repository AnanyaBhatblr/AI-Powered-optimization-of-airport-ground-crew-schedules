from flask import Flask
from routes.airport_routes import bp as airport_routes_bp


def create_app():
    app = Flask(__name__)

    # Configure app settings
    app.config.from_mapping(
        SECRET_KEY='dev',  # Change this to a secure key in production
    )

    # Register blueprints
    app.register_blueprint(airport_routes_bp)

    return app
