import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
from datetime import datetime
import random
import os

genai.configure(api_key="AIzaSyDYrvDTaHjeW6OYKPYa8V3eCZBQzVEs39M")
model = genai.GenerativeModel("gemini-1.5-flash")

# Define global variables
departments = [
    "Aircraft and Passenger Operations", "Baggage and Cargo Handling", 
    "Aircraft Servicing", "Safety and Maintenance Support", 
    "Equipment Operation and Maintenance", "Communication and Coordination", 
    "Emergency and Contingency Tasks"
]

# Update the terminal_to_folder mapping
terminal_to_folder = {
    "Bengaluru Terminal 1": "BT1",
    "Bengaluru Terminal 2": "BT2",
    "Delhi Terminal 1": "DT1",
    "Delhi Terminal 2": "DT2",
    "Delhi Terminal 3": "DT3",
    "Mumbai Terminal 1": "MT1",
    "Mumbai Terminal 2": "MT2",
    "Hyderabad Terminal 1": "HT1"
}
# Helper functions (unchanged)
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
        y=crew_feedback['task_completion_rate'],
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
def extract_numeric_id(crew_id):
    return int(crew_id.split('-')[1])  # Extract the numeric part after 'C-'

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
    performance_trends = feedback_df.groupby('date').agg({
        'task_completion_rate': 'mean',
        'quality_score': 'mean'
    }).reset_index()
    return performance_trends

def create_performance_visualization(performance_trends):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=performance_trends['date'],
        y=performance_trends['task_completion_rate'],
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
def load_data(terminal):
    base_path = terminal_to_folder.get(terminal)
    if not base_path:
        raise ValueError(f"Invalid terminal selection: {terminal}")
    
    # Construct the full path with Airport_Skill directory
    full_path = os.path.join(os.path.dirname(__file__), base_path)
    
    try:
        crew_df = pd.read_csv(f"{full_path}/crew_data.csv")
        performance_df = pd.read_csv(f"{full_path}/crew_performance_{base_path}.csv")
        feedback_df = pd.read_csv(f"{full_path}/performance_feedback.csv")
        training_df = pd.read_csv(f"{full_path}/training_history.csv")
        return crew_df, performance_df, feedback_df, training_df
    except FileNotFoundError as e:
        st.error(f"Could not load data files from {full_path}. Please check if all required files exist.")
        raise e
def main():

# Title of the app
    st.title("Crew Performance Analysis Dashboard")


# Sidebar (on the left)
    with st.sidebar:
        st.header("Filters")  # Sidebar header

    # Dropdown for terminal selection
        selected_location = st.selectbox(
            "City Terminal Number",
            [
                "Bengaluru Terminal 1", "Bengaluru Terminal 2", "Delhi Terminal 1", 
                "Delhi Terminal 2", "Delhi Terminal 3", "Mumbai Terminal 1", 
                "Mumbai Terminal 2", "Hyderabad Terminal 1"
            ]
        )

    # Dropdown for analysis type
        indiv_or_dept = st.selectbox(
            "Select Analysis Choice",
            ["Individual Crew Analysis", "Department-wise Analysis"]
        )


# Mapping of terminal names to folder names
# Load datasets based on the selected terminal


# Load datasets
    crew_df, performance_df, feedback_df, training_df = load_data(selected_location)


# Ensure consistent column names
    crew_df.rename(columns={"Crew_ID": "crew_id"}, inplace=True)
    performance_df.rename(columns={"Crew_ID": "crew_id"}, inplace=True)
    feedback_df.rename(columns={"Crew_ID": "crew_id"}, inplace=True)
    training_df.rename(columns={"Crew_ID": "crew_id"}, inplace=True)



# Main area of the app
    if indiv_or_dept == "Individual Crew Analysis":
    # Extract unique crew_ids for the selected location
    # Sort crew_ids numerically
        crew_ids = sorted(crew_df['crew_id'].unique().tolist(), key=extract_numeric_id)


    # Dropdown menu for crew_ids
        selected_crew_id = st.selectbox(
            "Select a Crew ID",
            crew_ids
        )


    # Display individual crew performance
        st.header(f"Performance Analysis for Crew ID: {selected_crew_id}")


    # Filter crew info for the selected crew ID
        crew_info = crew_df[crew_df['crew_id'] == selected_crew_id].iloc[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Department", performance_df[crew_df['crew_id'] == selected_crew_id].iloc[0]['department'])
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
        if not individual_training.empty:
            training_chart = px.scatter(
                individual_training, x='completion_date', y='score', color='training_module',
                title="Training History Timeline"
            )
            st.plotly_chart(training_chart)
        else:
            st.write("No training history available for this crew member.")


    # Performance History
        st.subheader("Performance History")
        crew_feedback = get_crew_performance_history(feedback_df, selected_crew_id)
        if not crew_feedback.empty:
            st.plotly_chart(create_performance_history_visualization(crew_feedback))
        else:
            st.write("No performance history available for this crew member.")


    # AI-Powered Individual Insights
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


    # Department-wise analysis
        st.header("Department-wise Analysis")


    # Dropdown for department selection
        selected_dept = st.selectbox(
            "Select a Department",
            departments
        )


    # Filter crew IDs for the selected department
        department_crew_ids = performance_df[performance_df['department'] == selected_dept]['crew_id'].unique()


    # Department Average Skills
        st.subheader("Department Average Skills")
    
    # Join crew_df with performance_df to get skill scores for the selected department
        department_crew_data = crew_df[crew_df['crew_id'].isin(department_crew_ids)]

        if not department_crew_data.empty:
        # Calculate average skill scores
            skill_columns = [col for col in department_crew_data.columns if col.endswith('_score')]
            avg_skills = department_crew_data[skill_columns].mean()
        
        # Visualize average skills
            avg_skills_chart = px.bar(
                avg_skills.reset_index(), x='index', y=0,
                labels={'index': 'Skill', 0: 'Average Score'}, title="Department Average Skills"
            )
            avg_skills_chart.update_layout(
                yaxis_title="Average Score"  # Explicitly set the y-axis title
            )
            st.plotly_chart(avg_skills_chart)
        else:
            st.write("No skill data available for this department.")

    # Department Training History
        st.subheader("Department Training History")
        department_training = training_df[training_df['crew_id'].isin(department_crew_ids)]
        if not department_training.empty:
            department_training_progress = analyze_training_progress(department_training)
            st.plotly_chart(create_training_progress_visualization(department_training_progress))
        else:
            st.write("No training history available for this department.")

    # Department Performance History
        st.subheader("Department Performance History")
        department_feedback = feedback_df[feedback_df['crew_id'].isin(department_crew_ids)]
        if not department_feedback.empty:
            department_performance_trends = analyze_performance_trends(department_feedback)
            st.plotly_chart(create_performance_visualization(department_performance_trends))
        else:
            st.write("No performance history available for this department.")


if __name__ == "__main__":
    main()