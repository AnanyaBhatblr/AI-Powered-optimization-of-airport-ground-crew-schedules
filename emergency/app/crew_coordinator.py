import pandas as pd
from datetime import datetime
import numpy as np
import random


class CrewCoordinator:
    def __init__(self):
        self.crews = None
        self.tasks = None
        self.available_crew = pd.DataFrame()
        self.active_incidents = pd.DataFrame()
        self.avg_speed = 50  # Average speed in km/h
        self.airport_locations = {}
        self._initialize_airport_locations()

    def load_data(self, crews_file, tasks_file):
        self.crews = pd.read_csv(crews_file)
        self.tasks = pd.read_csv(tasks_file)

    def load_crew_data(self, crew_file):
        """Load crew availability and skills data"""
        self.available_crew = pd.read_csv(crew_file)

        # Validate required columns
        required_columns = ['Crew_ID', 'Skill_Type',
                            'Current_Location', 'Availability']
        missing_columns = [
            col for col in required_columns if col not in self.available_crew.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in crew data: {
                             missing_columns}")

    def find_available_crew(self, task_time, task_location):
        """Find available crew members for a given time and location"""
        if self.crews is None:
            raise ValueError("Crew data not loaded")

        available = self.crews[
            ~self.crews['Crew_ID'].isin(
                self.tasks[
                    (self.tasks['Start_Time'] <= task_time) &
                    (self.tasks['End_Time'] >= task_time)
                ]['Crew_ID']
            )
        ]

        return available

    def get_random_crew(self):
        """Generate a random crew ID between 1 and 100"""
        crew_num = str(random.randint(1, 100)).zfill(4)
        return f"CRW{crew_num}"

    def find_nearest_crew(self, incident_location, required_skill):
        """Enhanced crew finding with random assignment"""
        # For now, just return a random crew member
        return {'Crew_ID': self.get_random_crew()}

    def reassign_crews(self, new_incident):
        """Reassign crews based on incident priority"""
        severity = new_incident['Severity']
        location = new_incident['Location']

        if severity >= 2:  # High or Critical severity
            # Pull crew from lower priority tasks
            return self.reassign_from_lower_priority(location)

        return self.find_nearest_crew(location, 'Skilled')

    def _calculate_distance(self, location1, location2):
        """Calculate Euclidean distance between two locations"""
        if location1 not in self.airport_locations or location2 not in self.airport_locations:
            return float('inf')  # Return infinity for invalid locations

        x1, y1 = self.airport_locations[location1]
        x2, y2 = self.airport_locations[location2]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _initialize_airport_locations(self):
        """Initialize the airport locations mapping"""
        self.airport_locations = {
            'Terminal 1': (0, 0),
            'Terminal 2': (500, 0),
            'Ground Services Hub': (200, -100),
            'Fire Station Main': (300, -800),
            'Fire Station Secondary': (300, 800),
            'MRO Facility': (-800, -600),
            # Add other locations as needed
        }
