import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import random

# Configure Gemini API
genai.configure(api_key="AIzaSyDYrvDTaHjeW6OYKPYa8V3eCZBQzVEs39M")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

def run_skill_analysis():
    st.title("Crew Performance Analysis Dashboard")
    
    try:
        # Load the dataset
        crew_df = pd.read_csv("Airport_final/crew_data.csv")
        training_df = pd.read_csv('Airport_final/training_history.csv')
        feedback_df = pd.read_csv('Airport_final/performance_feedback.csv')    
        performance_df = pd.read_csv("Airport_final/crew_performance_data.csv")

        # Extract unique values
        locations = sorted(performance_df['Location'].unique().tolist())
        departments = sorted(crew_df['department'].unique().tolist())

        # Sidebar filters
        with st.sidebar:
            st.header("Filters")
            
            indiv_or_dept = st.selectbox(
                "Select Analysis Choice",
                ["Individual Crew Analysis", "Department-wise Analysis"]
            )
            
            if indiv_or_dept == "Individual Crew Analysis":
                selected_crew_id = st.selectbox("Select Crew ID", sorted(crew_df['crew_id'].unique()))
            else:
                selected_dept = st.selectbox("Select Department", departments)
                filtered_dept = crew_df[crew_df['department'] == selected_dept]

        # Display based on selection
        if indiv_or_dept == "Individual Crew Analysis":
            # Display individual crew performance
            st.header(f"Performance Analysis for Crew ID: {selected_crew_id}")
            
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
            # Individual Training History
            st.subheader("Individual Training History")
            individual_training = training_df[training_df['crew_id'] == selected_crew_id]
            training_chart = px.scatter(
                individual_training, x='completion_date', y='score', color='training_module',
                title="Training History Timeline"
            )
            st.plotly_chart(training_chart)
            # Performance History
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
            # Display department-wise analysis
            st.header(f"Analysis for Department: {selected_dept}")
            # Get crew IDs for the selected department
            department_crew_ids = crew_df[crew_df['department'] == selected_dept]['crew_id'].unique()
            # Department Average Skills
            st.subheader("Department Average Skills")
            avg_skills = filtered_dept.drop(['crew_id', 'join_date', 'department', 'experience_years'], axis=1).mean()
            avg_skills_chart = px.bar(
                avg_skills.reset_index(), x='index', y=0,
                labels={'index': 'Skill', 0: 'Average Score'}, title="Department Average Skills"
            )
            st.plotly_chart(avg_skills_chart)
            # Department Training History
            st.subheader("Department Training History")
            department_training = training_df[training_df['crew_id'].isin(department_crew_ids)]
            department_training_progress = analyze_training_progress(department_training)
            st.plotly_chart(create_training_progress_visualization(department_training_progress))
            # Department Performance History
            st.subheader("Department Performance History")
            department_feedback = feedback_df[feedback_df['crew_id'].isin(department_crew_ids)]
            department_performance_trends = analyze_performance_trends(department_feedback)
            st.plotly_chart(create_performance_visualization(department_performance_trends))
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return

# Keep all helper functions outside run_skill_analysis()
def analyze_skill_gaps(crew_df):
    skill_columns = [col for col in crew_df.columns if col.endswith('_score')]
    skill_gaps = pd.DataFrame({
        'Skill': [col.replace('_score', '').replace('_', ' ').title() for col in skill_columns],
        'Average Score': [crew_df[col].mean() for col in skill_columns]
    })
    return skill_gaps
def get_crew_member_skills(crew_df, crew_id):
    skill_columns = [col for col in crew_df.columns if col.endswith('_score')]
    crew_skills = pd.DataFrame({
        'Skill': [col.replace('_score', '').replace('_', ' ').title() for col in skill_columns],
        'Score': [crew_df[crew_df['crew_id'] == crew_id][col].iloc[0] for col in skill_columns]
    })
    return crew_skills
def create_skill_comparison_visualization(crew_skills, avg_skills):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Crew Member',
        x=crew_skills['Skill'],
        y=crew_skills['Score'],
        marker_color='rgb(26, 118, 255)'
    ))
    fig.add_trace(go.Bar(
        name='Department Average',
        x=avg_skills['Skill'],
        y=avg_skills['Average Score'],
        marker_color='rgba(26, 118, 255, 0.3)'
    ))
    fig.update_layout(
        title='Individual vs Department Average Skills',
        barmode='group',
        xaxis_tickangle=-45,
        height=500,
        yaxis_title='Score'
    )
    return fig

def create_performance_history_visualization(crew_feedback):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=crew_feedback['date'],
        y=crew_feedback['task_completion_rate'],  # Already in percentage
        name='Task Completion Rate (%)',
        line=dict(color='rgb(26, 118, 255)')
    ))
    
    fig.add_trace(go.Scatter(
        x=crew_feedback['date'],
        y=crew_feedback['quality_score'],
        name='Quality Score',
        line=dict(color='rgb(56, 168, 0)')
    ))
    
    fig.update_layout(
        title='Individual Performance History',
        xaxis_title='Date',
        yaxis_title='Score',
        height=500
    )
    return fig
def get_crew_performance_history(feedback_df, crew_id):
    crew_feedback = feedback_df[feedback_df['crew_id'] == crew_id].sort_values('date')
    return crew_feedback

def get_gemini_insights_for_crew(crew_data, training_data, feedback_data):
    prompt = f"""
    Analyze the following crew member data:
    
    Crew Profile:
    {crew_data}
    
    Recent Training:
    {training_data}
    
    Performance Feedback:
    {feedback_data}
    
    Please provide:
    1. Strengths and areas for improvement
    2. Specific training recommendations
    3. Performance trends and suggestions
    
    Format the response in clear, concise bullets.
    """
    response = model.generate_content(prompt)
    return response.text

def analyze_training_progress(training_df):
    training_df['completion_date'] = pd.to_datetime(training_df['completion_date'])
    training_progress = training_df.groupby(['training_module', 'completion_date'])['score'].mean().reset_index()
    return training_progress
def create_training_progress_visualization(training_progress):
    fig = px.line(training_progress, 
                  x='completion_date', 
                  y='score', 
                  color='training_module',
                  title='Training Progress Over Time')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Score',
        height=500
    )
    return fig

def analyze_performance_trends(feedback_df):
    feedback_df['date'] = pd.to_datetime(feedback_df['date'])
    
    # Task completion rate is already in percentage (0 to 100)
    performance_trends = feedback_df.groupby('date').agg({
        'task_completion_rate': 'mean',
        'quality_score': 'mean'
    }).reset_index()
    return performance_trends

def create_performance_visualization(performance_trends):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_trends['date'],
        y=performance_trends['task_completion_rate'],  # Already in percentage
        name='Task Completion Rate (%)',
        line=dict(color='rgb(26, 118, 255)')
    ))
    
    fig.add_trace(go.Scatter(
        x=performance_trends['date'],
        y=performance_trends['quality_score'],
        name='Quality Score',
        line=dict(color='rgb(56, 168, 0)')
    ))
    
    fig.update_layout(
        title='Department Performance Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Score',
        height=500
    )
    return fig

if __name__ == "__main__":
    run_skill_analysis()
