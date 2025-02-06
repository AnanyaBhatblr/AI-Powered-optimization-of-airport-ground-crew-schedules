import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Import modules from existing files
# Update the import path to match the actual directory name
from Airport_final.final_skill import (
    analyze_skill_gaps, get_crew_member_skills, create_skill_comparison_visualization,
    create_performance_history_visualization, get_crew_performance_history,
    get_gemini_insights_for_crew, analyze_training_progress,
    create_training_progress_visualization, analyze_performance_trends,
    create_performance_visualization
)

from finaloperations.lstmapp import (
    plot_3d_bar_chart, display_dataset_info, plot_task_distribution
)

# Configure page
st.set_page_config(page_title="Airport Ground Crew Management System", layout="wide")

# Main title
st.title("Airport Ground Crew Management System")

# Create tabs for different functionalities
tab1, tab2 = st.tabs(["Crew Performance Analysis", "Task Allocation & Prediction"])

# Tab 1: Crew Performance Analysis
with tab1:
    # Load datasets
    crew_df = pd.read_csv("Airport final/crew_data.csv")
    training_df = pd.read_csv('Airport final/training_history.csv')
    feedback_df = pd.read_csv('Airport final/performance_feedback.csv')    
    performance_df = pd.read_csv("Airport final/crew_performance_data.csv")

    # Extract unique values
    locations = sorted(performance_df['Location'].unique().tolist())
    departments = sorted(crew_df['department'].unique().tolist())

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        analysis_choice = st.selectbox(
            "Select Analysis Choice",
            ["Individual Crew Analysis", "Department-wise Analysis"]
        )

        if analysis_choice == "Individual Crew Analysis":
            selected_location = st.selectbox("Select a Location", locations)
            filtered_crew = performance_df[performance_df['Location'] == selected_location]
            crew_ids = sorted(filtered_crew['crew_id'].unique().tolist())
            selected_crew_id = st.selectbox("Select a Crew ID", crew_ids)
        else:
            selected_dept = st.selectbox("Select a Department", departments)
            filtered_dept = crew_df[crew_df['department'] == selected_dept]

    # Main content area
    if analysis_choice == "Individual Crew Analysis":
        st.header(f"Performance Analysis for Crew ID: {selected_crew_id}")

        crew_info = crew_df[crew_df['crew_id'] == selected_crew_id].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Department", crew_info['department'])
        with col2:
            st.metric("Experience (Years)", f"{crew_info['experience_years']:.1f}")
        with col3:
            st.metric("Join Date", crew_info['join_date'])

        st.subheader("Skills Analysis")
        crew_skills = get_crew_member_skills(crew_df, selected_crew_id)
        avg_skills = analyze_skill_gaps(crew_df)
        st.plotly_chart(create_skill_comparison_visualization(crew_skills, avg_skills))

        st.subheader("Individual Training History")
        individual_training = training_df[training_df['crew_id'] == selected_crew_id]
        training_chart = px.scatter(
            individual_training, x='completion_date', y='score', color='training_module',
            title="Training History Timeline"
        )
        st.plotly_chart(training_chart)

        st.subheader("Performance History")
        crew_feedback = get_crew_performance_history(feedback_df, selected_crew_id)
        st.plotly_chart(create_performance_history_visualization(crew_feedback))

        st.subheader("AI-Powered Individual Insights")
        if st.button("Generate Individual Insights"):
            with st.spinner("Analyzing crew member data..."):
                individual_insights = get_gemini_insights_for_crew(
                    crew_info.to_dict(),
                    individual_training.tail().to_dict(),
                    crew_feedback.tail().to_dict()
                )
                st.write(individual_insights)

    else:
        st.header(f"Analysis for Department: {selected_dept}")
        department_crew_ids = crew_df[crew_df['department'] == selected_dept]['crew_id'].unique()

        st.subheader("Department Average Skills")
        avg_skills = filtered_dept.drop(['crew_id', 'join_date', 'department', 'experience_years'], axis=1).mean()
        avg_skills_chart = px.bar(
            avg_skills.reset_index(), x='index', y=0,
            labels={'index': 'Skill', 0: 'Average Score'}, title="Department Average Skills"
        )
        st.plotly_chart(avg_skills_chart)

        st.subheader("Department Training History")
        department_training = training_df[training_df['crew_id'].isin(department_crew_ids)]
        department_training_progress = analyze_training_progress(department_training)
        st.plotly_chart(create_training_progress_visualization(department_training_progress))

        st.subheader("Department Performance History")
        department_feedback = feedback_df[feedback_df['crew_id'].isin(department_crew_ids)]
        department_performance_trends = analyze_performance_trends(department_feedback)
        st.plotly_chart(create_performance_visualization(department_performance_trends))

# Tab 2: Task Allocation & Prediction
with tab2:
    # Load task allocation data
    crew_data = pd.read_csv("finaloperations/crew_dataDT1.csv")
    predictions_df = pd.read_csv("finaloperations/predDT1.csv")

    st.header("Task Allocation Analysis")

    # Display dataset information
    display_dataset_info(crew_data)

    # Plot task distribution
    plot_task_distribution(crew_data)

    # Optimize and visualize task allocation
    if st.button("Generate Task Allocation Visualization"):
        with st.spinner("Optimizing task allocation..."):
            plot_3d_bar_chart(predictions_df)