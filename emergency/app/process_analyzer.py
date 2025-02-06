import pandas as pd


class ProcessAnalyzer:
    def __init__(self):
        self.incident_history = pd.DataFrame()
        self.response_times = {}
        self.resolved_incidents = pd.DataFrame()
        # Add response_metrics initialization
        self.response_metrics = {}

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

    def analyze_resolution_times(self):
        """Analyze resolution times of resolved incidents"""
        # Check both incident history and resolved incidents
        resolved = self.incident_history[self.incident_history['Status'] == 'Resolved']

        if resolved.empty:
            return {
                "No Data": {
                    'avg_resolution_time': 0,
                    'incident_count': 0,
                    'by_location': {}
                }
            }

        analysis = {}
        # Group by severity and location
        for severity in resolved['Severity'].unique():
            severity_data = resolved[resolved['Severity'] == severity]

            analysis[f"Severity {severity}"] = {
                'avg_resolution_time': round(severity_data['Resolution_Time'].mean(), 2),
                'incident_count': len(severity_data),
                'by_location': severity_data.groupby('Location')['Resolution_Time'].mean().to_dict()
            }

        return analysis

    def get_incidents_by_location(self, include_resolved=True):
        """Get incident counts by location for heatmap"""
        if self.incident_history is None or self.incident_history.empty:
            return pd.Series(dtype=int)

        incidents_data = self.incident_history if include_resolved else \
            self.incident_history[self.incident_history['Status'] == 'Active']

        if 'Location' not in incidents_data.columns:
            return pd.Series(dtype=int)

        return incidents_data.groupby('Location').size()

    def update_response_metrics(self, incident_id, resolution_time):
        """Update response time metrics when an incident is resolved"""
        try:
            if incident_id in self.incident_history['Incident_ID'].values:
                # Update the incident in our history
                mask = self.incident_history['Incident_ID'] == incident_id
                self.incident_history.loc[mask, 'Status'] = 'Resolved'
                self.incident_history.loc[mask, 'Resolution_Time'] = resolution_time

                # Get incident details for metrics
                incident = self.incident_history[mask].iloc[0]

                # Create response metric
                metric = {
                    'incident_id': incident_id,
                    'location': incident['Location'],
                    'severity': str(incident['Severity']),  # Convert severity to string
                    'resolution_time': resolution_time,
                    'estimated_time': incident.get('Estimated_Resolution_Time', 0)
                }

                # Store in response metrics
                severity = str(incident['Severity'])  # Convert severity to string
                if severity not in self.response_metrics:
                    self.response_metrics[severity] = []
                self.response_metrics[severity].append(metric)

                return True
            return False
        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
            return False
