HYDERABAD_AIRPORT_LOCATIONS = {
    # Terminals
    'Terminal 1': (0, 0),    # Integrated passenger terminal
    'Terminal 2': (600, 0),  # Future expansion

    # Gates
    'Gate 1': (-50, 30),
    'Gate 2': (-30, 30),
    'Gate 3': (0, 30),
    'Gate 4': (30, 30),
    'Gate 20': (570, 30),
    'Gate 21': (600, 30),
    'Gate 22': (630, 30),

    # Runways and Taxiways
    'Runway 09L-27R': (1000, -1200),  # Main runway 4260m
    'Runway 09R-27L': (1000, 1200),   # Parallel runway
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

HYDERABAD_CONNECTIONS = {
    ('Terminal 1', 'Gate 1'): 1,
    ('Terminal 1', 'Gate 2'): 1,
    ('Terminal 2', 'Gate 20'): 1,
    ('Terminal 2', 'Gate 21'): 1
}

HYDERABAD_RESTRICTED_AREAS = [
    [(0, 0), (100, 0), (100, 100), (0, 100)]  # Example restricted area
]
