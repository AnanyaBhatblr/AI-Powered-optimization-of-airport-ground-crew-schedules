import streamlit as st
import pandas as pd
from emergency.app.incident_router import IncidentRouter
from emergency.app.incident_manager import IncidentManager
from emergency.app.process_analyzer import ProcessAnalyzer
from emergency.app.crew_coordinator import CrewCoordinator
from emergency.app.airport_configs import AIRPORTS

def main():
    # Airport Selection in Sidebar
    st.sidebar.title("Airport Configuration")
    selected_airport = st.sidebar.selectbox(
        "Select Airport",
        options=list(AIRPORTS.keys()),
        format_func=lambda x: AIRPORTS[x]['name']
    )

    # Get selected airport configuration
    airport_config = AIRPORTS[selected_airport]
    airport_locations = airport_config['locations']

    # Display current airport info
    st.title(f"{airport_config['name']} ({airport_config['code']})")

    if airport_locations is None:  # For BLR (Bengaluru)
        airport_locations = {
            # Terminals with accurate dimensions
            'Terminal 1': (0, 0),  # Reference point
            'Terminal 2': (500, 0),  # 500m east of T1

            # Gates - Actual spacing
            'Gate 1': (-50, 30),
            'Gate 2': (-30, 30),
            'Gate 3': (0, 30),
            'Gate 4': (30, 30),
            'Gate 20': (470, 30),
            'Gate 21': (500, 30),
            'Gate 22': (530, 30),

            # Runways - Actual dimensions and spacing
            'Runway 09R-27L': (1000, -1200),  # South Runway
            'Runway 09L-27R': (1000, 1200),   # North Runway (2.4km separation)
            'Taxiway A': (1000, -600),        # Parallel to runways
            'Taxiway B': (1000, 600),
            'Taxiway C': (0, 0),              # Connecting taxiway

            # Ground Operations - Real distances
            'Ground Services Hub': (200, -100),
            'De-icing Station': (-200, -300),
            'Fuel Farm': (800, -400),
            'Cargo Complex': (-600, -500),

            # Maintenance & Support
            'MRO Facility': (-800, -600),
            'Maintenance Hangar': (-1000, -600),
            'Air India MRO': (-1200, -600),

            # Emergency & Support - Strategic locations
            'Fire Station Main': (300, -800),     # Near south runway
            'Fire Station Secondary': (300, 800),  # Near north runway
            'ATC Tower': (100, 0),                # Central location
            'Emergency Response Center': (150, -200),

            # Cargo & Logistics - Actual cargo terminal location
            'Cargo Terminal 1': (-700, -500),
            'Express Cargo': (-800, -500),
            'Air Mail Building': (-900, -500),

            # Support Facilities
            'Catering Unit': (600, -200),
            'Equipment Storage': (400, -300),
            'Vehicle Pool': (300, -300),
            'Waste Management': (700, -400)
        }

    # Define the severity levels dictionary properly
    SEVERITY_LEVELS = {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Critical"
    }

    # Initialize components
    incident_router = IncidentRouter(airport_locations)
    incident_manager = IncidentManager()
    process_analyzer = ProcessAnalyzer()
    crew_coordinator = CrewCoordinator()

    # Set process analyzer in incident manager
    incident_manager.set_process_analyzer(process_analyzer)

    # Load data
    crew_coordinator.load_crew_data('emergency/data/crew.csv')
    process_analyzer.load_incident_history('emergency/data/incidents.csv')

    # Path Visualization with updated locations
    st.header("Path Visualization")
    col1, col2 = st.columns(2)

    # Update location options based on selected airport
    location_options = sorted(list(airport_config['locations'].keys()))

    with col1:
        start_point = st.selectbox(
            "Start Location",
            options=location_options,
            help="Select starting point for route calculation"
        )
    with col2:
        end_point = st.selectbox(
            "End Location",
            options=location_options,
            help="Select destination point for route calculation"
        )

    if st.button("Find Path", key="find_path_button"):
        try:
            # Initialize router with current airport configuration
            incident_router = IncidentRouter(airport_locations)
            path, distance = incident_router.find_shortest_path(
                start_point, end_point)

            if path:
                st.success(f"Path found! Distance: {distance:.2f} meters")
                st.write("Route:", " → ".join(path))
                fig = incident_router.plot_path(path)
                st.pyplot(fig)
            else:
                st.error("No valid path exists between these locations!")
        except Exception as e:
            st.error(f"Error finding path: {str(e)}")

    # Incident Reporting with updated locations
    st.header("Incident Reporting")
    with st.expander("Report New Incident"):
        incident_text = st.text_area("Incident Description")
        incident_location = st.selectbox(
            "Location",
            options=location_options,
            help="Select incident location"
        )

        if st.button("Submit Incident"):
            if incident_text and incident_location:
                # Analyze incident
                analysis = incident_manager.analyze_incident_report(incident_text)
                severity_level = analysis.get('severity', 1)
                severity_text = SEVERITY_LEVELS.get(severity_level, "Unknown")

                # Create incident data
                incident_data = {
                    'Incident_ID': f"INC{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                    'Report_Text': incident_text,
                    'Location': incident_location,
                    'Severity': severity_level,
                    'Status': 'Active',
                    'Timestamp': pd.Timestamp.now(),
                    'Crew_Assigned': None,
                    'Resolution_Time': None
                }

                # Add incident
                incident_manager.add_incident(incident_data)

                st.success(f"Incident reported with severity: {severity_text}")

                # Find and assign crew
                nearest_crew = crew_coordinator.find_nearest_crew(
                    incident_location, "Skilled")
                if nearest_crew is not None:
                    incident_data['Crew_Assigned'] = nearest_crew['Crew_ID']
                    st.success(f"Assigned crew member: {nearest_crew['Crew_ID']}")
                else:
                    st.warning("No available crew members found")
            else:
                st.error("Please provide incident description and location")

    # Incident Analytics
    st.header("Incident Analytics")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Show Incident Heatmap"):
            incidents_by_location = process_analyzer.incident_history.groupby(
                'Location').size()
            st.bar_chart(incidents_by_location)

    with col2:
        if st.button("Analyze Response Times"):
            bottlenecks = process_analyzer.analyze_bottlenecks()
            st.write("Response Time Analysis:", bottlenecks)

    # Incident Management
    st.header("Incident Management")
    with st.expander("Incident Dashboard"):
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=list(incident_manager.severity_levels.values())
            )
        with col2:
            location_filter = st.multiselect(
                "Filter by Location",
                options=location_options
            )

        # Display active incidents
        st.subheader("Active Incidents")
        active_incidents = process_analyzer.get_active_incidents()
        if not active_incidents.empty:
            st.dataframe(active_incidents)
        else:
            st.info("No active incidents")

if __name__ == "__main__":
    main()
