class AirportGraph:
    def __init__(self):
        # Define precise locations based on actual Bengaluru Airport layout
        # Coordinates are in meters relative to airport reference point
        self.locations = {
            # Main Terminal Areas
            'T1': (0, 0),
            'T1_GATE_N': (-50, 50),    # Terminal 1 North Gates
            'T1_GATE_S': (-50, -50),   # Terminal 1 South Gates
            'T2': (800, -100),
            'T2_GATE_N': (750, -50),   # Terminal 2 North Gates
            'T2_GATE_S': (750, -150),  # Terminal 2 South Gates

            # Runways with Entry/Exit Points
            'RWY09_27': (-500, -1000),
            'RWY09_ENTRY': (-800, -1000),
            'RWY27_EXIT': (-200, -1000),
            'RWY09R_27L': (500, -1000),
            'RWY09R_ENTRY': (200, -1000),
            'RWY27L_EXIT': (800, -1000),

            # Aprons with Multiple Access Points
            'APRON1_N': (100, -150),
            'APRON1_S': (100, -250),
            'APRON1_C': (100, -200),
            'APRON2_N': (700, -250),
            'APRON2_S': (700, -350),
            'APRON2_C': (700, -300),
            'CARGO_APRON_E': (-550, 200),
            'CARGO_APRON_W': (-650, 200),

            # Taxiways with Intersections
            'TWY_A1': (-300, -600),
            'TWY_A2': (-300, -400),
            'TWY_B1': (300, -600),
            'TWY_B2': (300, -400),
            'TWY_C1': (-100, -800),
            'TWY_C2': (100, -800),

            # Service Areas
            'ATC': (0, -500),
            'CARGO': (-800, 300),
            'MRO': (1000, 200),
            'GSE1': (-200, -100),
            'GSE2': (900, -200),
            'FIRE_STN1': (-100, -400),
            'FIRE_STN2': (600, -800)
        }

        # Updated comprehensive connections with realistic paths
        self.connections = {
            # Terminal area connections
            'Terminal 1': ['Gate 1', 'Gate 2', 'Gate 3', 'Gate 4', 'Ground Services Hub', 'Taxiway A', 'ATC Tower'],
            'Terminal 2': ['Gate 20', 'Gate 21', 'Gate 22', 'Equipment Storage', 'Taxiway B'],

            # Gate interconnections
            'Gate 1': ['Terminal 1', 'Gate 2', 'Ground Services Hub'],
            'Gate 2': ['Gate 1', 'Gate 3', 'Terminal 1'],
            'Gate 3': ['Gate 2', 'Gate 4', 'Terminal 1'],
            'Gate 4': ['Gate 3', 'Terminal 1', 'Taxiway A'],
            'Gate 20': ['Terminal 2', 'Gate 21', 'Taxiway B'],
            'Gate 21': ['Gate 20', 'Gate 22', 'Terminal 2'],
            'Gate 22': ['Gate 21', 'Terminal 2', 'Equipment Storage'],

            # Runway and taxiway system
            'Runway 09R-27L': ['Taxiway A', 'Taxiway C', 'Fire Station Main'],
            'Runway 09L-27R': ['Taxiway B', 'Taxiway C', 'Fire Station Secondary'],
            'Taxiway A': ['Terminal 1', 'Gate 4', 'Runway 09R-27L', 'Taxiway C', 'Ground Services Hub'],
            'Taxiway B': ['Terminal 2', 'Gate 20', 'Runway 09L-27R', 'Taxiway C', 'Equipment Storage'],
            'Taxiway C': ['Taxiway A', 'Taxiway B', 'Cargo Complex', 'MRO Facility'],

            # Ground operations area
            'Ground Services Hub': [
                'Terminal 1', 'Gate 1', 'Taxiway A', 'De-icing Station',
                'Fire Station Main', 'Vehicle Pool', 'Equipment Storage'
            ],
            'De-icing Station': ['Ground Services Hub', 'Maintenance Hangar', 'Taxiway A'],
            'Fuel Farm': ['Equipment Storage', 'Vehicle Pool', 'Fire Station Secondary', 'Taxiway B'],
            'Cargo Complex': ['Taxiway C', 'Cargo Terminal 1', 'Express Cargo', 'Air Mail Building'],

            # Maintenance and technical areas
            'MRO Facility': ['Taxiway C', 'Maintenance Hangar', 'Air India MRO'],
            'Maintenance Hangar': ['MRO Facility', 'De-icing Station', 'Air India MRO'],
            'Air India MRO': ['MRO Facility', 'Maintenance Hangar', 'Taxiway C'],

            # Emergency and support facilities
            'Fire Station Main': ['Ground Services Hub', 'Runway 09R-27L', 'Emergency Response Center', 'Taxiway A'],
            'Fire Station Secondary': ['Taxiway B', 'Runway 09L-27R', 'Emergency Response Center'],
            'ATC Tower': ['Terminal 1', 'Emergency Response Center', 'Ground Services Hub'],
            'Emergency Response Center': ['ATC Tower', 'Fire Station Main', 'Fire Station Secondary'],

            # Cargo facilities
            'Cargo Terminal 1': ['Cargo Complex', 'Express Cargo', 'Air Mail Building'],
            'Express Cargo': ['Cargo Terminal 1', 'Cargo Complex', 'Air Mail Building'],
            'Air Mail Building': ['Cargo Terminal 1', 'Express Cargo', 'Cargo Complex'],

            # Support facilities
            'Catering Unit': ['Equipment Storage', 'Terminal 2', 'Vehicle Pool'],
            'Equipment Storage': ['Terminal 2', 'Ground Services Hub', 'Vehicle Pool', 'Catering Unit', 'Fuel Farm'],
            'Vehicle Pool': ['Ground Services Hub', 'Equipment Storage', 'Waste Management'],
            'Waste Management': ['Vehicle Pool', 'Equipment Storage']
        }

        self.restricted_areas = {
            'Runway 09R-27L': ['RUNWAY_ACCESS'],
            'Runway 09L-27R': ['RUNWAY_ACCESS'],
            'ATC Tower': ['ATC_ACCESS'],
            'Fuel Farm': ['HAZMAT_ACCESS']
        }

        # Define operational characteristics
        self.area_characteristics = {
            'T1': {
                'capacity': 20000000,  # Annual passenger capacity
                'gates': range(1, 13),
                'security_level': 'high'
            },
            'T2': {
                'capacity': 25000000,
                'gates': range(14, 30),
                'security_level': 'high'
            },
            'RWY09_27': {
                'length': 4000,  # meters
                'width': 45,     # meters
                'surface': 'concrete'
            },
            'RWY09R_27L': {
                'length': 4000,
                'width': 45,
                'surface': 'concrete'
            }
        }

    def get_all_connected_locations(self):
        """Return all locations that have connections"""
        all_locations = set(self.connections.keys())
        for targets in self.connections.values():
            all_locations.update(targets)
        return all_locations

    def get_neighbors(self, location):
        """Get all neighboring locations for a given location"""
        return self.connections.get(location, [])
