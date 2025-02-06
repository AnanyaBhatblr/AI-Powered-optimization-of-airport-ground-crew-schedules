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

# Define the crew_location_mapping used earlier
locations = ['BT1', 'BT2', 'HT1', 'MT1', 'MT2', 'DT1', 'DT2', 'DT3']
crew_ids = [f'C-{i+1:03d}' for i in range(1, 250)]
crew_location_mapping = {crew_id: random.choice(locations) for crew_id in crew_ids}

def load_data():
    crew_df = pd.read_csv('crew_data.csv')
    training_df = pd.read_csv('training_history.csv')
    feedback_df = pd.read_csv('performance_feedback.csv')
    performance_df = pd.read_csv('crew_performance_data.csv')
    
    # Add Location to crew_df based on crew_location_mapping
    crew_df['Location'] = crew_df['crew_id'].map(crew_location_mapping)
    
    return performance_df, crew_df, training_df, feedback_df

def analyze_skill_gaps(crew_df):
    skill_columns = [col for col in crew_df.columns if col.endswith('_score')]
    skill_gaps = pd.DataFrame({
        'Skill': [col.replace('_score', '').replace('_', ' ').title() for col in skill_columns],
        'Average Score': [crew_df[col].mean() for col in skill_columns]
    })
    return skill_gaps

def create_skill_gap_visualization(skill_gaps):
    fig = px.bar(skill_gaps, 
                 x='Skill', 
                 y='Average Score',
                 title='Department Skill Gap Analysis',
                 color='Average Score',
                 color_continuous_scale='viridis')
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title='Average Score',
        showlegend=False,
        height=500
    )
    return fig

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

def get_crew_training_history(training_df, crew_id):
    crew_training = training_df[training_df['crew_id'] == crew_id].sort_values('completion_date')
    return crew_training

def create_training_timeline(crew_training):
    fig = px.scatter(crew_training,
                    x='completion_date',
                    y='score',
                    color='training_module',
                    title='Individual Training History Timeline',
                    hover_data=['feedback'])
    fig.update_layout(
        xaxis_tickangle=-45,
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

def main():
    st.title("Crew Performance Analysis Dashboard")
    
    try:
        # Load data
        performance_df, crew_df, training_df, feedback_df = load_data()
        
        # Sidebar for filtering
        st.sidebar.header("Analysis Options")
        
        # Location selection
        selected_location = st.sidebar.selectbox("Select Location", locations)
        
        # Filter crew_df based on selected location
        filtered_crew_df = crew_df[crew_df['Location'] == selected_location]
        
        # Department selection (only one department can be selected)
        selected_department = st.sidebar.selectbox(
            "Select Department",
            options=filtered_crew_df['department'].unique()
        )
        
        # Filter data based on selected department
        department_crew_df = filtered_crew_df[filtered_crew_df['department'] == selected_department]
        department_crew_ids = department_crew_df['crew_id'].unique()
        
        # Department-level analysis
        st.header(f"Department Analysis - {selected_department}")
        
        # Department Average Skills
        st.subheader("Department Average Skills")
        department_skill_gaps = analyze_skill_gaps(department_crew_df)
        st.plotly_chart(create_skill_gap_visualization(department_skill_gaps))
        
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
        
        # Individual crew member analysis (unchanged)
        st.header("Individual Crew Analysis")
        
        # Crew ID selection (only show crew IDs for the selected location and department)
        all_crew_ids = sorted(department_crew_df['crew_id'].unique())
        
        # Use session state to retain selected_crew_id
        if 'selected_crew_id' not in st.session_state:
            st.session_state.selected_crew_id = all_crew_ids[0] if all_crew_ids else None
        
        selected_crew_id = st.sidebar.selectbox(
            "Select Crew ID", 
            all_crew_ids,
            index=all_crew_ids.index(st.session_state.selected_crew_id) if st.session_state.selected_crew_id in all_crew_ids else 0
        )
        
        # Update session state with the selected crew ID
        st.session_state.selected_crew_id = selected_crew_id
        
        if selected_crew_id:
            # Display crew member info
            crew_info = department_crew_df[department_crew_df['crew_id'] == selected_crew_id].iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Department", crew_info['department'])
            with col2:
                st.metric("Experience (Years)", f"{crew_info['experience_years']:.1f}")
            with col3:
                st.metric("Join Date", crew_info['join_date'])
            
            # Skills comparison
            st.subheader("Skills Analysis")
            crew_skills = get_crew_member_skills(department_crew_df, selected_crew_id)
            avg_skills = analyze_skill_gaps(department_crew_df)
            st.plotly_chart(create_skill_comparison_visualization(crew_skills, avg_skills))
            
            # Training history
            st.subheader("Individual Training History")
            crew_training = get_crew_training_history(training_df, selected_crew_id)
            st.plotly_chart(create_training_timeline(crew_training))
            
            # Performance history
            st.subheader("Performance History")
            crew_feedback = get_crew_performance_history(feedback_df, selected_crew_id)
            st.plotly_chart(create_performance_history_visualization(crew_feedback))
            
            # AI Insights for individual
            st.subheader("AI-Powered Individual Insights")
            if st.button("Generate Individual Insights"):
                with st.spinner("Analyzing crew member data..."):
                    individual_insights = get_gemini_insights_for_crew(
                        crew_info.to_dict(),
                        crew_training.tail().to_dict(),
                        crew_feedback.tail().to_dict()
                    )
                    st.write(individual_insights)
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required data files are present and properly formatted.")

if __name__ == "__main__":
    main()