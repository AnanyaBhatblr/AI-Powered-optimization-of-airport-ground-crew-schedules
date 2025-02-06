import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime

# Configure Gemini API
genai.configure(api_key="AIzaSyDYrvDTaHjeW6OYKPYa8V3eCZBQzVEs39M")  # Replace with your actual API key
model = genai.GenerativeModel("gemini-1.5-flash")

def load_data():
    crew_df = pd.read_csv('crew_data.csv')
    training_df = pd.read_csv('training_history.csv')
    feedback_df = pd.read_csv('performance_feedback.csv')
    return crew_df, training_df, feedback_df

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
                 title='Team Skill Gap Analysis',
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
    
    # Convert task_completion_rate to percentage (0 to 100)
    feedback_df['task_completion_rate'] = feedback_df['task_completion_rate'] * 100
    
    performance_trends = feedback_df.groupby('date').agg({
        'task_completion_rate': 'mean',
        'quality_score': 'mean'
    }).reset_index()
    return performance_trends

def create_performance_visualization(performance_trends):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_trends['date'],
        y=performance_trends['task_completion_rate']*100,
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
        title='Team Performance Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Score',
        height=500
    )
    return fig

def create_performance_history_visualization(crew_feedback):
    fig = go.Figure()
    
    # Convert task_completion_rate to percentage (0 to 100)
    crew_feedback['task_completion_rate'] = crew_feedback['task_completion_rate'] * 100
    
    fig.add_trace(go.Scatter(
        x=crew_feedback['date'],
        y=crew_feedback['task_completion_rate'],  # Now in percentage
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
        name='Team Average',
        x=avg_skills['Skill'],
        y=avg_skills['Average Score'],
        marker_color='rgba(26, 118, 255, 0.3)'
    ))
    
    fig.update_layout(
        title='Individual vs Team Average Skills',
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
                    title='Training History Timeline',
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

def create_performance_history_visualization(crew_feedback):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=crew_feedback['date'],
        y=crew_feedback['task_completion_rate'],
        name='Task Completion Rate',
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
        crew_df, training_df, feedback_df = load_data()
        
        # Sidebar for filtering
        st.sidebar.header("Analysis Options")
        
        # Crew ID selection
        all_crew_ids = sorted(crew_df['crew_id'].unique())
        selected_crew_id = st.sidebar.selectbox("Select Crew ID", ["All"] + all_crew_ids)
        
        # Department filter
        selected_department = st.sidebar.multiselect(
            "Select Department",
            options=crew_df['department'].unique(),
            default=crew_df['department'].unique()
        )
        
        # Main content
        if selected_crew_id == "All":
            # Team-level analysis
            filtered_crew = crew_df[crew_df['department'].isin(selected_department)]
            
            st.header("Team Skills Gap Analysis")
            skill_gaps = analyze_skill_gaps(filtered_crew)
            st.plotly_chart(create_skill_gap_visualization(skill_gaps))
            
            st.header("Team Training Progress")
            training_progress = analyze_training_progress(training_df)
            st.plotly_chart(create_training_progress_visualization(training_progress))
            
            st.header("Team Performance Trends")
            performance_trends = analyze_performance_trends(feedback_df)
            st.plotly_chart(create_performance_visualization(performance_trends))
            
        else:
            # Individual crew member analysis
            st.header(f"Individual Analysis - Crew ID: {selected_crew_id}")
            
            # Display crew member info
            crew_info = crew_df[crew_df['crew_id'] == selected_crew_id].iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Department", crew_info['department'])
            with col2:
                st.metric("Experience (Years)", f"{crew_info['experience_years']:.1f}")
            with col3:
                st.metric("Join Date", crew_info['join_date'])
            
            # Skills comparison
            st.subheader("Skills Analysis")
            crew_skills = get_crew_member_skills(crew_df, selected_crew_id)
            avg_skills = analyze_skill_gaps(crew_df)
            st.plotly_chart(create_skill_comparison_visualization(crew_skills, avg_skills))
            
            # Training history
            st.subheader("Training History")
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