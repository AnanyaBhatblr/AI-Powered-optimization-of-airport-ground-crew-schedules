from flask import Flask
from .delhi_airport import DELHI_AIRPORT_LOCATIONS, DELHI_CONNECTIONS, DELHI_RESTRICTED_AREAS
from .mumbai_airport import MUMBAI_AIRPORT_LOCATIONS, MUMBAI_CONNECTIONS, MUMBAI_RESTRICTED_AREAS
from .hyderabad_airport import HYDERABAD_AIRPORT_LOCATIONS, HYDERABAD_CONNECTIONS, HYDERABAD_RESTRICTED_AREAS

AIRPORTS = {
    'DEL': {
        'name': 'Indira Gandhi International Airport',
        'code': 'DEL',
        'locations': DELHI_AIRPORT_LOCATIONS,
        'connections': DELHI_CONNECTIONS,
        'restricted_areas': DELHI_RESTRICTED_AREAS
    },
    'BOM': {
        'name': 'Chhatrapati Shivaji Maharaj International Airport',
        'code': 'BOM',
        'locations': MUMBAI_AIRPORT_LOCATIONS,
        'connections': MUMBAI_CONNECTIONS,
        'restricted_areas': MUMBAI_RESTRICTED_AREAS
    },
    'HYD': {
        'name': 'Rajiv Gandhi International Airport',
        'code': 'HYD',
        'locations': HYDERABAD_AIRPORT_LOCATIONS,
        'connections': HYDERABAD_CONNECTIONS,
        'restricted_areas': HYDERABAD_RESTRICTED_AREAS
    },
    'BLR': {
        'name': 'Kempegowda International Airport',
        'code': 'BLR',
        'locations': {
            # Define Bengaluru airport locations
            'Terminal 1': (0, 0),
            'Terminal 2': (500, 0),
            'Gate 1': (-50, 30),
            'Gate 2': (-30, 30),
            'Gate 3': (0, 30),
            'Gate 4': (30, 30),
            'Gate 20': (470, 30),
            'Gate 21': (500, 30),
            'Gate 22': (530, 30),
            'Runway 09R-27L': (1000, -1200),
            'Runway 09L-27R': (1000, 1200),
            'Taxiway A': (1000, -600),
            'Taxiway B': (1000, 600),
            'Taxiway C': (0, 0),
            'Ground Services Hub': (200, -100),
            'De-icing Station': (-200, -300),
            'Fuel Farm': (800, -400),
            'Equipment Storage': (400, -300),
            'Vehicle Pool': (300, -300),
            'MRO Facility': (-800, -600),
            'Maintenance Hangar': (-1000, -600),
            'Air India MRO': (-1200, -600),
            'Fire Station Main': (300, -800),
            'Fire Station Secondary': (300, 800),
            'ATC Tower': (100, 0),
            'Emergency Response Center': (150, -200),
            'Cargo Complex': (-600, -500),
            'Cargo Terminal 1': (-700, -500),
            'Express Cargo': (-800, -500),
            'Air Mail Building': (-900, -500),
            'Catering Unit': (600, -200),
            'Waste Management': (700, -400)
        },
        'connections': {
            # Define connections
        },
        'restricted_areas': {
            # Define restricted areas
        }
    }
}


def create_app():
    app = Flask(__name__)

    # Configure app settings
    app.config.from_mapping(
        SECRET_KEY='dev',  # Change this to a secure key in production
    )

    # Register blueprints and routes here (can be expanded later)
    app.register_blueprint(routes.bp)

    return app
