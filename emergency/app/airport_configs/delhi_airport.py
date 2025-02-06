DELHI_AIRPORT_LOCATIONS = {
    # Terminals
    'Terminal 1': (0, 0),
    'Terminal 2': (500, 0),

    # Gates
    'Gate 1': (-50, 30),
    'Gate 2': (-30, 30),
    'Gate 3': (0, 30),
    'Gate 4': (30, 30),
    'Gate 20': (470, 30),
    'Gate 21': (500, 30),
    'Gate 22': (530, 30),

    # Runways and Taxiways
    'Runway 09R-27L': (1000, -1200),
    'Runway 09L-27R': (1000, 1200),
    'Taxiway A': (1000, -600),
    'Taxiway B': (1000, 600),
    'Taxiway C': (0, 0),

    # Ground Operations
    'Ground Services Hub': (200, -100),
    'De-icing Station': (-200, -300),
    'Fuel Farm': (800, -400),
    'Equipment Storage': (400, -300),
    'Vehicle Pool': (300, -300),

    # Maintenance & Support
    'MRO Facility': (-800, -600),
    'Maintenance Hangar': (-1000, -600),
    'Air India MRO': (-1200, -600),

    # Emergency & Support
    'Fire Station Main': (300, -800),
    'Fire Station Secondary': (300, 800),
    'ATC Tower': (100, 0),
    'Emergency Response Center': (150, -200),

    # Cargo & Logistics
    'Cargo Complex': (-600, -500),
    'Cargo Terminal 1': (-700, -500),
    'Express Cargo': (-800, -500),
    'Air Mail Building': (-900, -500),

    # Support Facilities
    'Catering Unit': (600, -200),
    'Waste Management': (700, -400)
}

DELHI_CONNECTIONS = {
    # Define paths between locations
    ('Terminal 1', 'Gate 1'): 1,
    ('Terminal 1', 'Gate 2'): 1,
    # Add more connections as needed
}

DELHI_RESTRICTED_AREAS = [
    # Define restricted area polygons
    [(0, 0), (100, 0), (100, 100), (0, 100)]  # Example restricted area
]
