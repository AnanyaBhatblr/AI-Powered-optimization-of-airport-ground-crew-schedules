import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import os

# Import functions from final_skill.py
from Airport_final.final_skill import (
    analyze_skill_gaps, get_crew_member_skills, create_skill_comparison_visualization,
    create_performance_history_visualization, get_crew_performance_history,
    get_gemini_insights_for_crew, analyze_training_progress,
    create_training_progress_visualization, analyze_performance_trends,
    create_performance_visualization
)

# Import functions from lstmapp.py
from finaloperations.lstmapp import (
    plot_3d_bar_chart, display_dataset_info, plot_task_distribution,
    optimize_task_allocation, prepare_data, train_and_evaluate_model
)

# Configure page
st.set_page_config(page_title="Airport Ground Crew Management System", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Crew Performance Analysis", "Crew Demand Prediction"])

if page == "Crew Performance Analysis":
    st.title("Crew Performance Analysis Dashboard")
    
    # Load datasets
    try:
        crew_df = pd.read_csv("Airport_final/crew_data.csv")
        training_df = pd.read_csv('Airport_final/training_history.csv')
        feedback_df = pd.read_csv('Airport_final/performance_feedback.csv')    
        performance_df = pd.read_csv("Airport_final/crew_performance_data.csv")

        # Extract unique values
        locations = sorted(performance_df['Location'].unique().tolist())
        departments = sorted(crew_df['department'].unique().tolist())

        # Analysis choice
        analysis_choice = st.sidebar.selectbox(
            "Select Analysis Choice",
            ["Individual Crew Analysis", "Department-wise Analysis"]
        )

        if analysis_choice == "Individual Crew Analysis":
            selected_location = st.sidebar.selectbox("Select a Location", locations)
            filtered_crew = performance_df[performance_df['Location'] == selected_location]
            crew_ids = sorted(filtered_crew['crew_id'].unique().tolist())
            selected_crew_id = st.sidebar.selectbox("Select a Crew ID", crew_ids)

            # Display individual analysis
            st.header(f"Performance Analysis for Crew ID: {selected_crew_id}")
            
            # Display crew info
            crew_info = crew_df[crew_df['crew_id'] == selected_crew_id].iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Department", crew_info['department'])
            with col2:
                st.metric("Experience (Years)", f"{crew_info['experience_years']:.1f}")
            with col3:
                st.metric("Join Date", crew_info['join_date'])

            # Skills Analysis
            st.subheader("Skills Analysis")
            crew_skills = get_crew_member_skills(crew_df, selected_crew_id)
            avg_skills = analyze_skill_gaps(crew_df)
            st.plotly_chart(create_skill_comparison_visualization(crew_skills, avg_skills))

            # Training History
            st.subheader("Training History")
            individual_training = training_df[training_df['crew_id'] == selected_crew_id]
            st.plotly_chart(create_training_progress_visualization(
                analyze_training_progress(individual_training)
            ))

            # Performance History
            st.subheader("Performance History")
            crew_feedback = get_crew_performance_history(feedback_df, selected_crew_id)
            st.plotly_chart(create_performance_history_visualization(crew_feedback))

        else:
            selected_dept = st.sidebar.selectbox("Select a Department", departments)
            filtered_dept = crew_df[crew_df['department'] == selected_dept]
            
            # Department Analysis
            st.header(f"Analysis for Department: {selected_dept}")
            department_crew_ids = filtered_dept['crew_id'].unique()

            # Department Skills
            st.subheader("Department Skills Overview")
            dept_skills = analyze_skill_gaps(filtered_dept)
            st.plotly_chart(create_skill_comparison_visualization(dept_skills, dept_skills))

            # Department Training
            st.subheader("Department Training Progress")
            dept_training = training_df[training_df['crew_id'].isin(department_crew_ids)]
            st.plotly_chart(create_training_progress_visualization(
                analyze_training_progress(dept_training)
            ))

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

else:
    st.title("Crew Demand Prediction And Dynamic Crew Reallocation")
    
    # Terminal selection
    terminals = [
        "Bengaluru Terminal 1", "Bengaluru Terminal 2", "Delhi Terminal 1", 
        "Delhi Terminal 2", "Delhi Terminal 3", "Mumbai Terminal 1", 
        "Mumbai Terminal 2", "Hyderabad Terminal 1"
    ]
    
    selected_terminal = st.selectbox("Select Airport Terminal", terminals)
    
    # File paths mapping
    terminal_paths = {
        "Bengaluru Terminal 1": ("dataset/BT1/predBT1.csv", "dataset/BT1/crew_dataBT1.csv"),
        "Bengaluru Terminal 2": ("dataset/BT2/predBT2.csv", "dataset/BT2/crew_dataBT2.csv"),
        # ... add other terminals
    }
    
    if selected_terminal in terminal_paths:
        pred_path, crew_path = terminal_paths[selected_terminal]
        
        if os.path.exists(pred_path) and os.path.exists(crew_path):
            predictions_df = pd.read_csv(pred_path)
            crew_data = pd.read_csv(crew_path)
            
            # Display dataset information
            display_dataset_info(crew_data)
            
            # Task distribution
            plot_task_distribution(crew_data)
            
            # Optimization and visualization
            if st.button("Generate Task Allocation"):
                with st.spinner("Optimizing task allocation..."):
                    optimized_allocation = optimize_task_allocation(predictions_df)
                    
                    # Display visualizations
                    plot_3d_bar_chart(optimized_allocation)
                    
                    # Download option
                    csv = optimized_allocation.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Allocation Results",
                        data=csv,
                        file_name="task_allocation.csv",
                        mime="text/csv"
                    )
        else:
            st.error("Data files not found for selected terminal")

if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info("Airport Ground Crew Management System v1.0")