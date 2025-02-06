import streamlit as st
import sys
import os
st.set_page_config(
    page_title="Airport Operations Management System",
    page_icon="✈️",
    layout="wide"
)
# Add the module paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'finaloperations'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Productivity'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Airport_final'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'emergency'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Airport_Skill'))  # Add new path

# Import the modules
from finaloperations import lstmapp
from Productivity import final_productivity
from Airport_Skill.main import main as skill_analysis  # Updated import
from emergency.app.app import main as emergency_main

# Configure the page


# Create the navigation
def home_page():
    st.title("✈️ Welcome to Airport Ground Operations Management System")
    st.markdown("""
    ### An Integrated Solution for Airport Ground Operations
    
    This application provides comprehensive tools for:
    
    * 🎯 **Crew Demand Prediction & Dynamic Allocation**
    * 📊 **Crew Performance Analytics**
    * 🎓 **Crew Skill Analysis**
    * 🚨 **Emergency Response Management**
    
    Select a module from the sidebar to get started.
    """)
    
    # Add some visual elements
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.info("📈 **Crew Demand Prediction**\n\nPredict and optimize crew allocation across different terminals and shifts.")
    with col2:
        st.info("📊 **Crew Performance Analysis**\n\nAnalyze individual and team performance metrics with detailed insights.")
    with col3:
        st.info("🎓 **Crew Skill Analysis**\n\nAnalyze crew skills and training performance across departments.")
    with col4:
        st.info("🚨 **Emergency Response**\n\nManage and coordinate emergency responses across airport locations.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Home", "Crew Productivity Monitoring", "Dynamic Crew Reallocation", "Crew Skill Analysis", "Emergency Response"]
)

# Display the selected page
if page == "Home":
    home_page()
elif page == "Crew Productivity Monitoring":
    final_productivity.main()
elif page == "Dynamic Crew Reallocation":
    lstmapp.main()
elif page == "Crew Skill Analysis":
    skill_analysis()  # Updated function call
elif page=="Emergency Response":
    emergency_main()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Airport Operations Management System v1.0")