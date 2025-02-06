# crew_analyzer.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class EnhancedCrewAnalyzer:
    def __init__(self, df):
        self.le_dict = {}  # Initialize first
        self.df = self._preprocess_data(df)
        self._train_models()
        
    def _preprocess_data(self, df):
        # Convert and sort dates
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Create month features
        df['Month'] = df['Date'].dt.month
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        df['Month_Name'] = df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])
        df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)
        
        # Calculate performance metrics
        df['Tasks_Per_Hour'] = (df['Tasks_Completed'] / (df['Shift_Duration']/60)).round(2)
        df['Efficiency_Rate'] = ((1 - (df['Idle_Time'] / df['Shift_Duration'])) * 100).round(2)
        df['Productivity_Index'] = ((df['Tasks_Completed'] * (1 - df['Idle_Time']/df['Shift_Duration'])) / 
                                  (df['Fatigue_Level'] * (df['Safety_Incidents'] + 1))).round(2)
        df['Risk_Factor'] = ((df['Safety_Incidents'] + 1) * df['Fatigue_Level']).round(2)
        
        # Encode categorical features
        categorical_cols = ['Weather_Condition', 'Equipment_Used', 'Location', 'Crew_ID']
        for col in categorical_cols:
            self.le_dict[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.le_dict[col].fit_transform(df[col])
            
        
        
        df['Original_Tasks_Per_Hour'] = df['Tasks_Per_Hour']
        df['Original_Efficiency_Rate'] = df['Efficiency_Rate']
        
        # Scale features
        self.scaler = StandardScaler()
        cluster_features = ['Tasks_Per_Hour', 'Efficiency_Rate', 'Risk_Factor']
        df[cluster_features] = self.scaler.fit_transform(df[cluster_features])
        
        return df

    def _train_models(self):
        # Gradient Boosting Model
        features = ['Task_Completion_Time', 'Idle_Time', 'Shift_Duration',
                   'Fatigue_Level', 'Experience_Level', 'Breaks_Taken',
                   'Weather_Condition_encoded', 'Equipment_Used_encoded',
                   'Location_encoded']
        
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.gb_model.fit(self.df[features], self.df['Productivity_Index'])
        self.feature_importance = dict(zip(features, self.gb_model.feature_importances_))
        
        # K-Means Clustering
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.df['Performance_Cluster'] = self.kmeans.fit_predict(self.df[['Tasks_Per_Hour', 'Efficiency_Rate', 'Risk_Factor']])
        
        # Autoencoder for anomalies
        input_dim = 3
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(int(input_dim/2), activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)
        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        self.autoencoder.fit(self.df[['Tasks_Per_Hour', 'Efficiency_Rate', 'Risk_Factor']], 
                            self.df[['Tasks_Per_Hour', 'Efficiency_Rate', 'Risk_Factor']], 
                            epochs=50, batch_size=32, shuffle=True, verbose=0)

   
    def analyze_crew(self, crew_id):
        crew_data = self.df[self.df['Crew_ID'] == crew_id].copy()
        if crew_data.empty:
            return None
        
            # Anomaly detection (using scaled values)
        reconstructed = self.autoencoder.predict(crew_data[['Tasks_Per_Hour', 'Efficiency_Rate', 'Risk_Factor']])
        mse = np.mean(np.power(crew_data[['Tasks_Per_Hour', 'Efficiency_Rate', 'Risk_Factor']] - reconstructed, 2), axis=1)
        crew_data['Is_Anomaly'] = mse > np.percentile(mse, 95)
            
            # Calculate rankings using ORIGINAL values
        all_crews = self.df.groupby('Crew_ID').agg({
            'Productivity_Index': 'mean',
            'Original_Efficiency_Rate': 'mean',  # Changed from Efficiency_Rate
            'Risk_Factor': 'mean'
        }).reset_index()

        results = {
            'crew_data': crew_data,
            'cluster': crew_data['Performance_Cluster'].mode()[0],
            'avg_productivity': crew_data['Productivity_Index'].mean(),
            'avg_efficiency': crew_data['Original_Efficiency_Rate'].mean(),  # Changed
            'anomalies': crew_data['Is_Anomaly'].sum(),
            'monthly_trends': crew_data.groupby('Month_Name').agg({
                'Productivity_Index': 'mean',
                'Original_Efficiency_Rate': 'mean',  # Changed
                'Risk_Factor': 'mean'
            }),
            'rankings': {
                'Productivity Rank': int(all_crews['Productivity_Index'].rank(ascending=False, method='min')[all_crews['Crew_ID'] == crew_id].values[0]),
                'Efficiency Rank': int(all_crews['Original_Efficiency_Rate'].rank(ascending=False, method='min')[all_crews['Crew_ID'] == crew_id].values[0]),  # Changed
                'Safety Rank': int(all_crews['Risk_Factor'].rank(ascending=True, method='min')[all_crews['Crew_ID'] == crew_id].values[0])
            },
            'stats': {
                'Overall Productivity Index': crew_data['Productivity_Index'].mean().round(2),
                'Risk Factor': crew_data['Risk_Factor'].mean().round(2),
                'Average Tasks Per Hour': crew_data['Original_Tasks_Per_Hour'].mean().round(2),  # Changed
                'Experience Level (Years)': crew_data['Experience_Level'].mean().round(2),
                'Efficiency Rate (%)': crew_data['Original_Efficiency_Rate'].mean().round(2),  # Changed
                'Total Safety Incidents': int(crew_data['Safety_Incidents'].sum())
            }
        }
            
        return results
# Now correctly returns the results dictionary
def create_performance_distribution(analyzer, crew_id):
    """Distribution comparison with proper labels"""
    crew_data = analyzer.df[analyzer.df['Crew_ID'] == crew_id]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=analyzer.df['Productivity_Index'], 
        name='All Crews',
        marker_color='#CCCCCC',
        opacity=0.6
    ))
    fig.add_trace(go.Histogram(
        x=crew_data['Productivity_Index'],
        name=f'Crew {crew_id}',
        marker_color='#636EFA',
        opacity=0.8
    ))
    fig.update_layout(
        title_text='Productivity Distribution Comparison',
        xaxis_title_text='Productivity Index',
        yaxis_title_text='Frequency',
        bargap=0.2
    )
    return fig

# CORRECTED EFFICIENCY CALCULATION IN PREPROCESSING
def _preprocess_data(self, df):
    # ... other preprocessing code ...
    
    # Fixed efficiency calculation
    df['Efficiency_Rate'] = ((1 - (df['Idle_Time'] / df['Shift_Duration'])) * 100).round(2)
    
    # ... rest of preprocessing code ...

# MODIFIED COMPARATIVE ANALYSIS WITH NEW EFFICIENCY VISUALIZATION
def create_comparative_analysis(analyzer, crew_id):
    """Combined visualization with updated efficiency comparison"""
    crew_data = analyzer.df[analyzer.df['Crew_ID'] == crew_id]
    all_crews_avg = analyzer.df.groupby('Month_Name').agg({
        'Productivity_Index': 'mean',
        'Efficiency_Rate': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Productivity Timeline', 'Efficiency Comparison',
                       'Risk Factor Analysis', 'Experience vs Productivity'),
        vertical_spacing=0.15
    )
    
    # Productivity Timeline (unchanged)
    # ... existing productivity timeline code ...
    
    # NEW EFFICIENCY COMPARISON (Box plots instead of histograms)
    fig.add_trace(
        go.Box(y=analyzer.df['Efficiency_Rate'], name='All Crews',
              marker_color='gray', boxpoints='outliers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=crew_data['Efficiency_Rate'], name='Selected Crew',
              marker_color='blue', boxpoints='outliers'),
        row=1, col=2
    )
    
    # Risk Factor Analysis (unchanged)
    # ... existing risk factor code ...
    
    # Experience vs Productivity (unchanged)
    # ... existing experience vs productivity code ...
    
    # Update layout
    fig.update_layout(
        height=800, 
        showlegend=True, 
        title_text=f'Performance Analysis: Crew {crew_id}',
        boxmode='group'  # Grouped box plots
    )
    fig.update_yaxes(title_text="Efficiency Rate (%)", row=1, col=2)
    
    return fig

# MODIFIED WEATHER IMPACT ANALYSIS
def create_weather_impact_analysis(crew_data):
    """Weather impact visualization with box plots"""
    fig = px.box(crew_data,
        x='Weather_Condition',
        y='Efficiency_Rate',
        color='Weather_Condition',
        points="all",
        hover_data=['Date'],
        labels={
            'Weather_Condition': 'Weather Condition',
            'Efficiency_Rate': 'Efficiency Rate (%)'
        })
    fig.update_layout(
        title='Weather Impact on Efficiency',
        xaxis_title='Weather Condition',
        yaxis_title='Efficiency Rate (%)',
        showlegend=False
    )
    return fig

def create_temporal_analysis(crew_data):
    """Time series analysis with labels"""
    fig = make_subplots(rows=3, cols=1, 
        subplot_titles=(
            'Productivity Trend', 
            'Efficiency Rate', 
            'Risk Factor'
        ),
        vertical_spacing=0.1)
    
    # Productivity
    fig.add_trace(go.Scatter(
        x=crew_data['Date'], y=crew_data['Productivity_Index'],
        name='Productivity', line=dict(color='#636EFA')),
        row=1, col=1)
    
    # Efficiency
    fig.add_trace(go.Scatter(
        x=crew_data['Date'], y=crew_data['Efficiency_Rate'],
        name='Efficiency', line=dict(color='#00CC96')),
        row=2, col=1)
    
    # Risk Factor
    fig.add_trace(go.Bar(
        x=crew_data['Date'], y=crew_data['Risk_Factor'],
        name='Risk Factor', marker_color='#EF553B'),
        row=3, col=1)
    
    fig.update_layout(
        height=600,
        title_text='Temporal Performance Analysis',
        showlegend=False
    )
    fig.update_yaxes(title_text="Productivity Index", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Risk Factor", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def create_workload_analysis(crew_data):
    """Workload distribution visualization using original values"""
    fig = px.density_heatmap(
        crew_data,
        x='Shift_Duration',
        y='Original_Tasks_Per_Hour',  # Changed
        z='Original_Efficiency_Rate',  # Changed
        histfunc="avg",
        labels={
            'Shift_Duration': 'Shift Duration (minutes)',
            'Original_Tasks_Per_Hour': 'Tasks Per Hour',  # Changed
            'Original_Efficiency_Rate': 'Efficiency Rate (%)'  # Changed
        },
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        title='Workload Efficiency Analysis',
        xaxis_title='Shift Duration (minutes)',
        yaxis_title='Tasks Per Hour',
        coloraxis_colorbar_title='Efficiency (%)'
    )
    return fig
def get_recommendation(insight_title):
    recommendation_map = {
        'High Performer': "🚀 Top Performer Strategies:\n"
                         "- Assign complex/high-priority tasks\n"
                         "- Feature in leadership programs\n"
                         "- Document best practices",
        
        'Low Productivity': "⚠️ Performance Improvement Plan:\n"
                           "- Conduct workflow audit\n"
                           "- Implement mentorship program\n"
                           "- Review equipment allocation",
        
        'High Risk Crew': "🔴 Safety Critical Actions:\n"
                         "- Mandatory safety training\n"
                         "- Fatigue management review\n"
                         "- Daily safety check protocol",
        
        'Safety Leader': "🛡️ Safety Excellence:\n"
                        "- Nominate for safety awards\n"
                        "- Feature in safety campaigns\n"
                        "- Share best practices",
        
        'Needs Optimization': "🔧 Optimization Strategies:\n"
                             "- Time-motion studies\n"
                             "- Equipment maintenance review\n"
                             "- Shift pattern analysis",
        
        'Rising Fatigue': "😴 Fatigue Management:\n"
                         "- Adjust shift rotations\n"
                         "- Mandatory break reminders\n"
                         "- Stress management workshops"
    }
    return recommendation_map.get(insight_title, 
                                "📈 Monitor performance and review operational parameters")

def generate_enhanced_insights(results, analyzer):
    """Generate contextual insights with recommendations"""
    insights = []
    crew_data = results['crew_data']
    
    # Productivity insights
    base_productivity = analyzer.df['Productivity_Index'].median()
    prod_percentile = (crew_data['Productivity_Index'] > base_productivity).mean() * 100
    
    if prod_percentile >= 75:
        insights.append(('success', 'High Performer', 
                       f"Top {100 - (crew_data['Productivity_Index'].rank(pct=True).mean() * 100):.0f}% in productivity"))
    elif prod_percentile <= 25:
        insights.append(('error', 'Low Productivity', 
                       f"Bottom {crew_data['Productivity_Index'].rank(pct=True).mean() * 100:.0f}% in productivity"))

    # Safety insights with comparative analysis
    org_risk = analyzer.df['Risk_Factor'].mean()
    crew_risk = crew_data['Risk_Factor'].mean()
    risk_ratio = crew_risk/org_risk
    
    if risk_ratio > 1.3:
        insights.append(('error', 'High Risk Crew', 
                       f"Risk score {crew_risk:.1f} (Org avg: {org_risk:.1f})"))
    elif risk_ratio < 0.7:
        insights.append(('success', 'Safety Leader', 
                       f"Risk score {crew_risk:.1f} (Org avg: {org_risk:.1f})"))

    # Cluster-based insights
    cluster_mean = analyzer.df.groupby('Performance_Cluster')['Productivity_Index'].mean()
    cluster_rank = cluster_mean.rank(ascending=False)[results['cluster']]
    
    performance_labels = {
        1: ('success', 'Top Performer', "Exceeds expectations across metrics"),
        2: ('info', 'Standard Performer', "Meets baseline requirements"),
        3: ('warning', 'Needs Improvement', "Requires targeted interventions")
    }
    insights.append(performance_labels.get(cluster_rank, ('info', 'Performance Category', "")))
    
    # Fatigue trend analysis
    if len(crew_data) > 1:
        fatigue_trend = crew_data['Fatigue_Level'].diff().mean()
        if fatigue_trend > 0.5:
            insights.append(('warning', 'Rising Fatigue', 
                           f"Daily fatigue increase: {fatigue_trend:.2f} points"))
        elif fatigue_trend < -0.3:
            insights.append(('success', 'Improving Fatigue', 
                           f"Daily fatigue decrease: {abs(fatigue_trend):.2f} points"))
    
    # Anomaly detection insight
    if results['anomalies'] > 0:
        anomaly_rate = (results['anomalies']/len(crew_data))*100
        insights.append(('error' if anomaly_rate > 15 else 'warning', 
                       'Performance Anomalies',
                       f"{results['anomalies']} unusual days detected ({anomaly_rate:.1f}%)"))
        
    return insights

def create_equipment_analysis(crew_data):
    """Sunburst chart from newapp"""
    return px.sunburst(
        crew_data,
        path=['Equipment_Used', 'Weather_Condition'],
        values='Tasks_Completed',
        color='Efficiency_Rate',
        color_continuous_scale='RdYlGn'
    )
def create_comparative_analysis(analyzer, crew_id):
    """Combined visualization from both versions"""
    crew_data = analyzer.df[analyzer.df['Crew_ID'] == crew_id]
    all_crews_avg = analyzer.df.groupby('Month_Name').agg({
        'Productivity_Index': 'mean',
        'Efficiency_Rate': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Productivity Timeline', 'Efficiency Distribution',
                       'Risk Factor Analysis', 'Experience vs Productivity'),
        vertical_spacing=0.15
    )
    
    # Productivity Timeline
    crew_data = crew_data.sort_values('Date')
    fig.add_trace(
        go.Scatter(x=crew_data['Date'], y=crew_data['Productivity_Index'],
                  name='Crew Productivity', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=all_crews_avg['Month_Name'], y=all_crews_avg['Productivity_Index'],
                  name='All Crews Avg', line=dict(color='gray', dash='dot')),
        row=1, col=1
    )
    
    # Efficiency Distribution
    fig.add_trace(
        go.Box(y=analyzer.df['Efficiency_Rate'], name='All Crews',
              marker_color='gray', boxpoints='outliers'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=crew_data['Efficiency_Rate'], name='Selected Crew',
              marker_color='blue', boxpoints='outliers'),
        row=1, col=2
    )
    
    # Risk Factor Analysis
    fig.add_trace(
        go.Scatter(x=crew_data['Date'], y=crew_data['Risk_Factor'],
                  mode='lines+markers', name='Risk Trend',
                  line=dict(color='red')),
        row=2, col=1
    )
    
    # Experience vs Productivity
    fig.add_trace(
        go.Scatter(x=analyzer.df['Experience_Level'], y=analyzer.df['Productivity_Index'],
                  mode='markers', name='All Crews', marker=dict(color='gray', opacity=0.4)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=crew_data['Experience_Level'], y=crew_data['Productivity_Index'],
                  mode='markers', name='Selected Crew', marker=dict(color='blue')),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800, 
        showlegend=True, 
        title_text=f'Performance Analysis: Crew {crew_id}',
        boxmode='group'  # Grouped box plots
    )
    fig.update_yaxes(title_text="Efficiency Rate (%)", row=1, col=2)
    
    return fig

def main():
    # Configure page
    st.title("✈️ Airport Crew Performance Dashboard")
    
    # Load data
    df = pd.read_csv('Productivity/productivity.csv')
    df['Crew_ID'] = df['Crew_ID'].astype(str).str.strip()
    analyzer = EnhancedCrewAnalyzer(df)
    
    # Sidebar controls
    st.sidebar.header("🔍For Analysis")
    locations = sorted(df['Location'].unique())
    selected_loc = st.sidebar.selectbox("Select Airport Location:", locations)
    
    crews = df[df['Location'] == selected_loc]['Crew_ID'].unique()
    selected_crew = st.sidebar.selectbox("Select Crew ID:", sorted(crews))
    
    def add_spacer(lines=2):
        for _ in range(lines):
            st.write("")
    
    if st.sidebar.button("🚀 Analyze Crew Performance"):
        with st.spinner("Analyzing performance data..."):
            results = analyzer.analyze_crew(selected_crew)
            
            if not results:
                st.error("No data found for selected crew")
                return
                
            crew_data = results['crew_data']
            
            # ================== Main Dashboard ==================
            st.header(f"📊 Performance Overview: Crew {selected_crew}")
            add_spacer()
            
            # Performance Metrics Section
            st.subheader("Performance Metrics")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.markdown("**Overall Productivity Index**")
                st.markdown(f"<h2 style='color: #4CAF50;'>{results['stats']['Overall Productivity Index']}</h2>", 
                           unsafe_allow_html=True)
                add_spacer(1)
                
                st.markdown("**Risk Factor**")
                st.markdown(f"<h2 style='color: #F44336;'>{results['stats']['Risk Factor']}</h2>", 
                           unsafe_allow_html=True)
                
            with metrics_col2:
                st.markdown("**Average Tasks Per Hour**")
                st.markdown(f"<h2 style='color: #2196F3;'>{results['stats']['Average Tasks Per Hour']}</h2>", 
                           unsafe_allow_html=True)
                add_spacer(1)
                
                st.markdown("**Experience Level (Years)**")
                st.markdown(f"<h2 style='color: #9C27B0;'>{results['stats']['Experience Level (Years)']}</h2>", 
                           unsafe_allow_html=True)
                
            with metrics_col3:
                st.markdown("**Efficiency Rate (%)**")
                st.markdown(f"<h2 style='color: #FF9800;'>{results['stats']['Efficiency Rate (%)']}</h2>", 
                           unsafe_allow_html=True)
                add_spacer(1)
                
                st.markdown("**Total Safety Incidents**")
                st.markdown(f"<h2 style='color: #607D8B;'>{results['stats']['Total Safety Incidents']}</h2>", 
                           unsafe_allow_html=True)
            
            add_spacer(3)
            
            # Rankings Section
            st.subheader("Rankings")
            total_crews = len(analyzer.df['Crew_ID'].unique())
            rank_col1, rank_col2, rank_col3 = st.columns(3)
            
            with rank_col1:
                st.markdown("**Productivity Rank**")
                st.markdown(f"<h3 style='color: #4CAF50;'>{results['rankings']['Productivity Rank']} of {total_crews}</h3>", 
                           unsafe_allow_html=True)
                
            with rank_col2:
                st.markdown("**Efficiency Rank**")
                st.markdown(f"<h3 style='color: #2196F3;'>{results['rankings']['Efficiency Rank']} of {total_crews}</h3>", 
                           unsafe_allow_html=True)
                
            with rank_col3:
                st.markdown("**Safety Rank**")
                st.markdown(f"<h3 style='color: #F44336;'>{results['rankings']['Safety Rank']} of {total_crews}</h3>", 
                           unsafe_allow_html=True)
            
            add_spacer(3)
            
            # Core Visualizations
            st.subheader(" 📈 Comparative Performance" )
            fig = create_comparative_analysis(analyzer, selected_crew)
            st.plotly_chart(fig, use_container_width=True)
            add_spacer(3)
            
            # Temporal Analysis
            st.subheader("⏳ Historical Trends ")
            fig = create_temporal_analysis(crew_data)
            st.plotly_chart(fig, use_container_width=True)
            add_spacer(3)
            
            # Environmental Factors
            st.subheader("🍀 Environmental Impact Analysis")
            col3, col4 = st.columns(2)
            with col3:
                fig = create_weather_impact_analysis(crew_data)
                st.plotly_chart(fig, use_container_width=True)
            with col4:
                fig = create_workload_analysis(crew_data)
                st.plotly_chart(fig, use_container_width=True)
            add_spacer(3)
            
            # Equipment Analysis
            st.subheader("🔧 Equipment Utilization ")
            fig = create_equipment_analysis(crew_data)
            st.plotly_chart(fig, use_container_width=True)
            add_spacer(3)
            
            # Insights
            st.subheader("💡 Expert Recommendations")
            insights = generate_enhanced_insights(results, analyzer)
            
            for insight in insights:
                with st.expander(f"{insight[1]} - {insight[2]}", expanded=True):
                    if insight[0] == 'success':
                        st.success(get_recommendation(insight[1]))
                    elif insight[0] == 'error':
                        st.error(get_recommendation(insight[1]))
                    else:
                        st.info(get_recommendation(insight[1]))
            
            add_spacer(3)
            
            # Data Export
            st.subheader(" Export Results")
            csv = crew_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset",
                data=csv,
                file_name=f"crew_{selected_crew}_analysis.csv",
                mime='text/csv',
                help="Includes all raw data and calculated metrics for this analysis"
            )
            add_spacer(2)

if __name__ == "__main__":
    main()