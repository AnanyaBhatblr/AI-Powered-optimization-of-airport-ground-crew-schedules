import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import streamlit as st

# Initialize Streamlit app
st.title("Airport Ground Crew Performance Analysis")

# Load and preprocess data
@st.cache_data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Month_Name'] = df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])
    df['Tasks_Per_Hour'] = (df['Tasks_Completed'] / (df['Shift_Duration'] / 60)).round(2)
    df['Efficiency_Rate'] = ((1 - (df['Idle_Time'] / df['Shift_Duration'])) * 100).round(2)
    df['Productivity_Index'] = ((df['Tasks_Completed'] * (1 - df['Idle_Time'] / df['Shift_Duration'])) / 
                                (df['Fatigue_Level'] * (df['Safety_Incidents'] + 1))).round(2)
    df['Risk_Factor'] = ((df['Safety_Incidents'] + 1) * df['Fatigue_Level']).round(2)

    # Label encode categorical columns
    le_dict = {}
    categorical_cols = ['Weather_Condition', 'Equipment_Used', 'Location', 'Crew_ID']
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        df[f'{col}_encoded'] = le_dict[col].fit_transform(df[col])

    # Scale features
    scaler = StandardScaler()
    features = ['Tasks_Per_Hour', 'Efficiency_Rate', 'Experience_Level', 
                'Productivity_Index', 'Risk_Factor', 'Breaks_Taken']
    scaled_features = scaler.fit_transform(df[features])
    
    return df, le_dict, scaled_features

# Load data
df, le_dict, scaled_features = load_data("crew_performance_data.csv")

# Train models
@st.cache_data
def train_models(df, scaled_features):
    # Gradient Boosting
    features = ['Task_Completion_Time', 'Idle_Time', 'Shift_Duration', 
                'Fatigue_Level', 'Experience_Level', 'Breaks_Taken',
                'Weather_Condition_encoded', 'Equipment_Used_encoded', 'Location_encoded']
    X = df[features]
    y = df['Productivity_Index']
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X, y)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    df['Performance_Cluster'] = cluster_labels

    # Autoencoder for anomaly detection
    input_dim = scaled_features.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(int(input_dim / 2), activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, shuffle=True, verbose=0)

    return gb_model, kmeans, autoencoder

# Train models
gb_model, kmeans, autoencoder = train_models(df, scaled_features)

# Crew selection
crew_ids = sorted(df['Crew_ID'].unique())
selected_crew = st.selectbox("Select Crew ID:", crew_ids)

# Analyze selected crew
def analyze_crew(selected_crew):
    crew_data = df[df['Crew_ID'] == selected_crew]

    # Calculate performance metrics
    crew_stats = {
        'Overall Productivity Index': crew_data['Productivity_Index'].mean().round(2),
        'Average Tasks Per Hour': crew_data['Tasks_Per_Hour'].mean().round(2),
        'Efficiency Rate (%)': crew_data['Efficiency_Rate'].mean().round(2),
        'Risk Factor': crew_data['Risk_Factor'].mean().round(2),
        'Experience Level (Years)': crew_data['Experience_Level'].mean().round(2),
        'Total Safety Incidents': crew_data['Safety_Incidents'].sum()
    }

    # Monthly trends
    monthly_trends = crew_data.groupby('Month_Name').agg({
        'Productivity_Index': 'mean',
        'Tasks_Per_Hour': 'mean',
        'Efficiency_Rate': 'mean',
        'Safety_Incidents': 'sum'
    })

    # Feature importance visualization
    st.markdown("### Feature Importance: Factors Affecting Performance")
    fig, ax = plt.subplots()
    features = ['Task_Completion_Time', 'Idle_Time', 'Shift_Duration', 
                'Fatigue_Level', 'Experience_Level', 'Breaks_Taken',
                'Weather_Condition_encoded', 'Equipment_Used_encoded', 'Location_encoded']
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': gb_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    importance_df.plot.barh(x='Feature', y='Importance', ax=ax, color='teal')
    ax.set_title('Factors Affecting Performance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Features')
    st.pyplot(fig)

    # Monthly trends visualization
    st.markdown("### Monthly Trends")
    fig, ax = plt.subplots()
    monthly_trends['Productivity_Index'].plot(kind='line', marker='o', ax=ax)
    ax.set_title(f'Monthly Productivity Trend for {selected_crew}')
    ax.set_xlabel('Month')
    ax.set_ylabel('Productivity Index')
    st.pyplot(fig)

    # Safety risk analysis
    st.markdown("### Safety Risk Analysis")
    fig, ax = plt.subplots()
    sns.scatterplot(data=crew_data, x='Fatigue_Level', y='Safety_Incidents', 
                    size='Risk_Factor', sizes=(50, 200), ax=ax)
    ax.set_title('Safety Risk Analysis')
    st.pyplot(fig)

    # Anomaly detection
    reconstructed = autoencoder.predict(scaled_features)
    mse = np.mean(np.power(scaled_features - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 95)
    crew_data['Is_Anomaly'] = mse > threshold
    anomalies = crew_data['Is_Anomaly'].sum()
    st.markdown(f"### Anomalies Detected: {anomalies} days")

# Display analysis
if selected_crew:
    analyze_crew(selected_crew)
