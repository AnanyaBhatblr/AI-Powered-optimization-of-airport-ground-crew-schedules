import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
from datetime import datetime, timedelta
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

def load_model(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# Load models
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
kmeans = load_model(os.path.join(models_dir, "kmeans_model.pkl"))
arima = load_model(os.path.join(models_dir, "arima_model.pkl"))

# App title and configuration
st.set_page_config(page_title="Crew Scheduler", layout="wide")
st.title("Crew Scheduler and Task Allocator")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", 
    ["Data Upload", "Task Clustering", "Demand Forecasting", "Schedule Optimization"])

# Global state for uploaded data
if 'tasks' not in st.session_state:
    st.session_state.tasks = None
if 'crew' not in st.session_state:
    st.session_state.crew = None
if 'disruptions' not in st.session_state:
    st.session_state.disruptions = None

# Data Upload Page
if page == "Data Upload":
    st.header("Data Upload")
    
    with st.expander("Upload Files", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            task_file = st.file_uploader("Upload Task Details CSV", type="csv")
            if task_file:
                st.session_state.tasks = pd.read_csv(task_file)
                
        with col2:
            crew_file = st.file_uploader("Upload Crew Details CSV", type="csv")
            if crew_file:
                st.session_state.crew = pd.read_csv(crew_file)
                
        with col3:
            disruption_file = st.file_uploader("Upload Disruption Details CSV", type="csv")
            if disruption_file:
                st.session_state.disruptions = pd.read_csv(disruption_file)

    if st.session_state.tasks is not None:
        with st.expander("Task Details"):
            st.dataframe(st.session_state.tasks)
            
    if st.session_state.crew is not None:
        with st.expander("Crew Details"):
            st.dataframe(st.session_state.crew)
            
    if st.session_state.disruptions is not None:
        with st.expander("Disruption Details"):
            st.dataframe(st.session_state.disruptions)

# Task Clustering Page
elif page == "Task Clustering":
    st.header("Task Clustering Analysis")
    
    if st.session_state.tasks is not None:
        with st.expander("Clustering Settings"):
            n_clusters = st.slider("Number of Clusters", 2, 5, 3)
            features = st.multiselect("Select Features for Clustering", 
                                    st.session_state.tasks.columns.tolist(),
                                    default=['Task Type', 'Task Duration'])
        
        if st.button("Run Clustering"):
            task_features = pd.get_dummies(st.session_state.tasks[features])
            clusters = kmeans.predict(task_features)
            st.session_state.tasks['Cluster'] = clusters
            st.success("Clustering completed!")
            st.write("Clustered Results:")
            st.dataframe(st.session_state.tasks)
    else:
        st.warning("Please upload task data first!")

# Demand Forecasting Page
elif page == "Demand Forecasting":
    st.header("Demand Forecasting")
    
    with st.expander("Forecast Settings"):
        forecast_period = st.selectbox("Forecast Period", ["Daily", "Weekly", "Monthly"])
        horizon = st.slider("Forecast Horizon", 1, 30, 7)
        
    if st.button("Generate Forecast"):
        forecast_dates = pd.date_range(start=datetime.now(), periods=horizon)
        forecast_values = arima.forecast(steps=horizon)
        
        st.line_chart(pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted Demand': forecast_values
        }).set_index('Date'))

# Schedule Optimization Page
elif page == "Schedule Optimization":
    st.header("Schedule Optimization")
    
    if st.session_state.tasks is not None and st.session_state.crew is not None:
        with st.expander("Optimization Settings"):
            objective = st.selectbox("Optimization Objective", 
                ["Minimize Total Time", "Maximize Efficiency", "Balance Workload"])
            constraints = st.multiselect("Additional Constraints",
                ["Break Times", "Skill Requirements", "Working Hours"])
        
        if st.button("Optimize Schedule"):
            predictions_df = st.session_state.tasks.copy()
            optimized_allocation = optimize_task_allocation(predictions_df)
            st.success("Schedule optimized!")
            st.write("Optimized Schedule:")
            st.dataframe(optimized_allocation)
    else:
        st.warning("Please upload both task and crew data first!")

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("v1.0.0 | Made with ❤️ by AI Team")

def optimize_task_allocation(predictions_df):
    # Create optimization model
    model = LpProblem("Task_Reallocation", LpMinimize)
    
    # Get unique values
    tasks = predictions_df['Task_ID'].unique()
    gates = predictions_df['Gate_number'].unique()
    shifts = predictions_df['Shift_no'].unique()
    
    # Decision variables
    assignments = LpVariable.dicts("assign",
                                 ((t, g, s) for t in tasks for g in gates for s in shifts),
                                 cat='Binary')
    
    # Objective: Minimize total predicted crew demand deviation
    model += lpSum(assignments[t, g, s] * predictions_df[
        (predictions_df['Task_ID'] == t) &
        (predictions_df['Gate_number'] == g) &
        (predictions_df['Shift_no'] == s)]['Predicted'].values[0]
                  for t in tasks for g in gates for s in shifts)
    
    # Constraints
    # Each task must be assigned once per shift
    for t in tasks:
        for s in shifts:
            model += lpSum(assignments[t, g, s] for g in gates) == 1
            
    # Maximum tasks per gate per shift
    for g in gates:
        for s in shifts:
            model += lpSum(assignments[t, g, s] for t in tasks) <= 3  # Adjust capacity as needed
    
    # Ensure tasks are only assigned where there is a need for crew allocation (positive values)
    for t in tasks:
        for g in gates:
            for s in shifts:
                predicted_value = predictions_df[
                    (predictions_df['Task_ID'] == t) &
                    (predictions_df['Gate_number'] == g) &
                    (predictions_df['Shift_no'] == s)
                ]['Predicted'].values[0]
                
                if predicted_value <= 0:
                    model += assignments[t, g, s] == 0
    
    # Solve
    model.solve()
    
    # Extract results
    results = []
    for t in tasks:
        for g in gates:
            for s in shifts:
                if value(assignments[t, g, s]) > 0:
                    # Get the number of free crews from predictions_df
                    free_crews = predictions_df[
                        (predictions_df['Task_ID'] == t) &
                        (predictions_df['Gate_number'] == g) &
                        (predictions_df['Shift_no'] == s)
                    ]['Predicted'].values[0]
                    
                    results.append({
                        'Task_ID': t,
                        'Gate_number': g,
                        'Shift_no': s,
                        'Allocated': free_crews
                    })
    
    return pd.DataFrame(results)