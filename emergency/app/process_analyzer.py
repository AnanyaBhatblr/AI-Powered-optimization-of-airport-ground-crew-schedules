import pandas as pd


class ProcessAnalyzer:
    def __init__(self):
        self.incident_history = pd.DataFrame()
        self.response_times = {}

    def load_incident_history(self, file_path):
        """Load historical incident data"""
        self.incident_history = pd.read_csv(file_path)
        self.incident_history['Timestamp'] = pd.to_datetime(
            self.incident_history['Timestamp'])

    def analyze_bottlenecks(self):
        """Identify bottlenecks in incident response"""
        avg_response_times = self.incident_history.groupby(
            'Location')['Resolution_Time'].mean()
        bottlenecks = avg_response_times[avg_response_times >
                                         avg_response_times.mean()]
        return bottlenecks

    def suggest_improvements(self):
        """Generate improvement suggestions based on analysis"""
        bottlenecks = self.analyze_bottlenecks()
        suggestions = []

        for location, response_time in bottlenecks.items():
            if response_time > 120:  # 2 hours
                suggestions.append(f"Critical: High response times at {
                                   location}. Consider additional crew allocation.")
            elif response_time > 60:  # 1 hour
                suggestions.append(f"Warning: Moderate delays at {
                                   location}. Review crew scheduling.")

        return suggestions

    def add_incident(self, incident_data):
        """Add new incident to history"""
        # Convert incident_data to DataFrame if it's a dictionary
        new_incident = pd.DataFrame([incident_data]) if isinstance(
            incident_data, dict) else incident_data

        # Use concat instead of append
        self.incident_history = pd.concat(
            [self.incident_history, new_incident], ignore_index=True)

    def get_active_incidents(self):
        """Get currently active incidents"""
        return self.incident_history[
            (self.incident_history['Status'] == 'Active') |
            (self.incident_history['Status'] == 'Pending')
        ]
