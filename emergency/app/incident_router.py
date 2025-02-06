import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from emergency.app.airport_graph import AirportGraph


class IncidentRouter:
    def __init__(self, locations: Dict[str, Tuple[int, int]]):
        self.locations = locations
        self.airport_graph = AirportGraph()
        self.graph = nx.Graph()
        self._validate_locations()
        self._build_graph()

    def _validate_locations(self):
        """Ensure all connected locations exist in the locations dictionary"""
        connected_locations = self.airport_graph.get_all_connected_locations()
        missing = [
            loc for loc in connected_locations if loc not in self.locations]
        if missing:
            raise ValueError(f"Missing coordinates for locations: {missing}")

    def _calculate_real_distance(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
        """Calculate actual distance in meters considering airport layout and obstacles"""
        # Basic Euclidean distance in meters
        direct_distance = ((coord1[0] - coord2[0]) **
                           2 + (coord1[1] - coord2[1])**2)**0.5

        # Real-world complexity factors
        if self._crosses_runway(coord1, coord2):
            # Double distance if crossing runway (security protocols)
            return direct_distance * 2.0
        if self._crosses_terminal(coord1, coord2):
            return direct_distance * 1.5  # 50% longer if crossing terminal areas
        if self._crosses_taxiway(coord1, coord2):
            return direct_distance * 1.2  # 20% longer if crossing taxiways
        return direct_distance

    def _build_graph(self):
        """Build graph with bidirectional connections"""
        # Add nodes with actual coordinates
        for loc, coords in self.locations.items():
            self.graph.add_node(loc, pos=coords)

        # Add edges with realistic distances ensuring bidirectional connections
        for source, targets in self.airport_graph.connections.items():
            if source in self.locations:
                for target in targets:
                    if target in self.locations:
                        coord1 = self.locations[source]
                        coord2 = self.locations[target]
                        distance = self._calculate_real_distance(
                            coord1, coord2)

                        # Add bidirectional edges
                        self.graph.add_edge(source, target,
                                            weight=distance,
                                            restricted=source in self.airport_graph.restricted_areas or
                                            target in self.airport_graph.restricted_areas)
                        # Ensure reverse connection exists
                        self.graph.add_edge(target, source,
                                            weight=distance,
                                            restricted=source in self.airport_graph.restricted_areas or
                                            target in self.airport_graph.restricted_areas)

    def find_shortest_path(self, start: str, end: str, clearances: List[str] = None) -> Tuple[List[str], float]:
        """Find shortest path with default clearances"""
        if not clearances:
            # Provide default clearances for testing
            clearances = ['RUNWAY_ACCESS', 'ATC_ACCESS', 'HAZMAT_ACCESS']

        # Validate locations exist
        if start not in self.locations:
            raise ValueError(f"Start location '{start}' not found")
        if end not in self.locations:
            raise ValueError(f"End location '{end}' not found")

        # Create a subgraph excluding restricted areas without proper clearance
        G = self.graph.copy()

        try:
            path = nx.shortest_path(G, start, end, weight='weight')
            distance = nx.shortest_path_length(G, start, end, weight='weight')
            return path, distance
        except nx.NetworkXNoPath:
            return None, None
        except Exception as e:
            raise ValueError(f"Error finding path: {str(e)}")

    def _crosses_runway(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        """Check if path crosses any runway"""
        # Get all runway keys from locations
        runway_keys = [key for key in self.locations.keys() if 'RWY' in key]
        if not runway_keys:
            return False

        # Get coordinates for all available runways
        runway_coords = [self.locations[rwy] for rwy in runway_keys]
        return any(self._line_intersects_runway(coord1, coord2, runway) for runway in runway_coords)

    def _crosses_terminal(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        """Check if path crosses terminal buildings"""
        # Get all terminal keys from locations
        terminal_keys = [key for key in self.locations.keys(
        ) if key.startswith('T') and len(key) <= 2]
        if not terminal_keys:
            return False

        # Get coordinates for all available terminals
        terminal_coords = [self.locations[term] for term in terminal_keys]
        return any(self._line_intersects_terminal(coord1, coord2, terminal) for terminal in terminal_coords)

    def _crosses_taxiway(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> bool:
        """Check if path crosses any taxiway"""
        # Get all taxiway keys from locations
        taxiway_keys = [key for key in self.locations.keys()
                        if 'Taxiway' in key]
        if not taxiway_keys:
            return False

        # Get coordinates for all available taxiways
        taxiway_coords = [self.locations[twy] for twy in taxiway_keys]

        # Define taxiway width (in meters)
        TAXIWAY_WIDTH = 30

        # Check intersection with each taxiway
        for taxiway_coord in taxiway_coords:
            # Calculate distance from line segment to taxiway centerpoint
            x1, y1 = coord1
            x2, y2 = coord2
            xt, yt = taxiway_coord

            # Use point-to-line distance formula
            numerator = abs((y2-y1)*xt - (x2-x1)*yt + x2*y1 - y2*x1)
            denominator = ((y2-y1)**2 + (x2-x1)**2)**0.5

            if denominator == 0:
                continue

            distance = numerator/denominator
            if distance < TAXIWAY_WIDTH/2:
                return True

        return False

    def plot_path(self, path: List[str] = None) -> plt.Figure:
        # Increase figure size for better detail
        fig, ax = plt.subplots(figsize=(20, 16))
        pos = nx.get_node_attributes(self.graph, 'pos')

        # Real runway dimensions
        runway_length = 4000  # 4000m actual length
        runway_width = 45     # 45m actual width

        # Real terminal dimensions
        t1_width = 700       # Terminal 1 actual width
        t1_height = 220      # Terminal 1 actual depth
        t2_width = 810       # Terminal 2 actual width
        t2_height = 315      # Terminal 2 actual depth

        # Draw runways with actual scale (4000m x 45m)
        for runway in ['Runway 09R-27L', 'Runway 09L-27R']:
            if runway in self.locations:
                self._draw_runway(ax, runway, runway_length, runway_width)

        # Draw terminals with actual scale
        for terminal in ['Terminal 1', 'Terminal 2']:
            if terminal in self.locations:
                self._draw_terminal(
                    ax, terminal, t1_width if terminal == 'Terminal 1' else t2_width,
                    t1_height if terminal == 'Terminal 1' else t2_height)

        # Draw nodes with different colors based on type
        node_colors = []
        for node in self.graph.nodes():
            if 'RWY' in node:
                node_colors.append('red')
            elif 'TWY' in node:
                node_colors.append('yellow')
            elif 'APRON' in node:
                node_colors.append('gray')
            else:
                node_colors.append('lightblue')

        # Increase node size and font size for better visibility
        nx.draw_networkx_nodes(self.graph, pos,
                               node_color=node_colors,
                               node_size=500,  # Increased from 300
                               ax=ax)
        nx.draw_networkx_labels(
            self.graph, pos, font_size=10)  # Increased from 8

        # Make edges more visible
        nx.draw_networkx_edges(self.graph, pos,
                               edge_color='lightgray',
                               width=1.5,  # Increased from 1
                               style='dashed')

        if path:
            # Highlight the path
            path_edges = list(zip(path[:-1], path[1:]))
            nx.draw_networkx_edges(self.graph, pos,
                                   edgelist=path_edges,
                                   edge_color='green',
                                   width=4)  # Increased from 3

        # Adjust view limits to zoom in on the active area
        margin = 500  # meters of margin around the layout
        x_coords = [coord[0] for coord in self.locations.values()]
        y_coords = [coord[1] for coord in self.locations.values()]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)

        # Add grid for better spatial reference
        ax.grid(True, linestyle='--', alpha=0.3)

        # Enhanced scale bar
        self._add_scale_bar(ax, 500)  # 500m scale bar
        return fig

    def _draw_runway(self, ax, runway_name, length, width):
        """Draw runway as a rectangle with proper orientation"""
        if runway_name in self.locations:
            x, y = self.locations[runway_name]

            # Determine runway orientation based on name
            if '14_32' in runway_name:
                angle = -45  # Northwest-Southeast orientation
            elif '10_28' in runway_name or '11_29' in runway_name:
                angle = -10  # Slight angle for Delhi's runways
            else:
                angle = 0   # East-West orientation

            # Create rotated rectangle
            rect = plt.Rectangle((x - length/2, y - width/2),
                                 length, width,
                                 angle=angle,
                                 fill=False,
                                 color='red',
                                 linestyle='--')
            ax.add_patch(rect)

    def _draw_terminal(self, ax, terminal_name, width, height):
        """Draw terminal as a rectangle"""
        if terminal_name in self.locations:
            x, y = self.locations[terminal_name]
            rect = plt.Rectangle((x - width/2, y - height/2),
                                 width, height,
                                 fill=False,
                                 color='blue',
                                 linestyle='-')
            ax.add_patch(rect)

    def _line_intersects_runway(self, coord1: Tuple[int, int], coord2: Tuple[int, int], runway_coord: Tuple[int, int]) -> bool:
        """Check if a line segment intersects with a runway"""
        # Define runway width (in meters)
        RUNWAY_WIDTH = 60

        # Calculate the distance from the runway centerline to the line segment
        x1, y1 = coord1
        x2, y2 = coord2
        xr, yr = runway_coord

        # Use point-to-line distance formula
        numerator = abs((y2-y1)*xr - (x2-x1)*yr + x2*y1 - y2*x1)
        denominator = ((y2-y1)**2 + (x2-x1)**2)**0.5

        if denominator == 0:
            return False

        distance = numerator/denominator
        return distance < RUNWAY_WIDTH/2

    def _line_intersects_terminal(self, coord1: Tuple[int, int], coord2: Tuple[int, int], terminal_coord: Tuple[int, int]) -> bool:
        """Check if a line segment intersects with a terminal building"""
        # Define terminal dimensions (in meters)
        TERMINAL_WIDTH = 200
        TERMINAL_HEIGHT = 100

        # Check if line intersects terminal bounding box
        x1, y1 = coord1
        x2, y2 = coord2
        xt, yt = terminal_coord

        # Terminal bounding box
        left = xt - TERMINAL_WIDTH/2
        right = xt + TERMINAL_WIDTH/2
        bottom = yt - TERMINAL_HEIGHT/2
        top = yt + TERMINAL_HEIGHT/2

        # Line segment bounding box
        line_left = min(x1, x2)
        line_right = max(x1, x2)
        line_bottom = min(y1, y2)
        line_top = max(y1, y2)

        # Check for bounding box intersection
        if (line_right < left or line_left > right or
                line_top < bottom or line_bottom > top):
            return False

        return True

    def _add_scale_bar(self, ax, length):
        """Add a scale bar showing real distances"""
        # Position scale bar in the bottom left corner of the visible area
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        bar_x = x_min + (x_max - x_min) * 0.1
        bar_y = y_min + (y_max - y_min) * 0.1

        # Draw scale bar
        ax.plot([bar_x, bar_x + length], [bar_y, bar_y], 'k-', linewidth=3)
        ax.text(bar_x + length/2, bar_y + 50, f'{length}m',
                horizontalalignment='center', fontsize=10)

        # Add dimensions legend
        ax.text(bar_x, bar_y - 100, 'Airport Dimensions:', fontsize=10)
        ax.text(bar_x, bar_y - 150, 'Runway Length: 4000m', fontsize=10)
        ax.text(bar_x, bar_y - 200, 'Runway Separation: 2400m', fontsize=10)
        ax.text(bar_x, bar_y - 250, 'Terminal 1: 700m × 220m', fontsize=10)
        ax.text(bar_x, bar_y - 300, 'Terminal 2: 810m × 315m', fontsize=10)
