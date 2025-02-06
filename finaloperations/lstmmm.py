import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import io
import streamlit as st
def plot_3d_bar_chart(optimized_allocation):
    st.subheader("3D Bar Plot of Optimized Task Allocation")

    # Convert Task_ID to numeric
    optimized_allocation['Task_ID_numeric'] = pd.factorize(optimized_allocation['Task_ID'])[0]

    # Extract numeric values
    x = optimized_allocation['Shift_no']
    y = optimized_allocation['Gate_number']
    z = optimized_allocation['Task_ID_numeric']
    allocated_values = optimized_allocation['Allocated']

    # Normalize bar width
    dx = dy = 0.5
    dz = allocated_values

    # Normalize colors based on Allocated values
    norm = plt.Normalize(allocated_values.min(), allocated_values.max())
    colors = plt.cm.YlOrRd(norm(allocated_values))

    # Plot the 3D bars
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(x, y, z, dx, dy, dz, color=colors)

    # Set labels and title
    ax.set_xlabel('Shift Number')
    ax.set_ylabel('Gate Number')
    ax.set_zlabel('Task ID (Numeric)')
    ax.set_title('3D Bar Plot of Optimized Task Allocation')

    # Add Task_ID labels with spacing to avoid overlap
    task_id_labels = optimized_allocation.drop_duplicates('Task_ID_numeric')[['Task_ID_numeric', 'Task_ID']]
    ax.set_zticks(task_id_labels['Task_ID_numeric'][::5])  # Show every 5th label
    ax.set_zticklabels(task_id_labels['Task_ID'][::5], fontsize=10, rotation=90)  # Adjust font and rotation

    # Add a color legend for Allocated values with integer ticks
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Allocated Value')

    # Set integer ticks for the colorbar
    int_ticks = np.arange(allocated_values.min(), allocated_values.max() + 1, 1)  # Ensure integer ticks
    cbar.set_ticks(int_ticks)
    cbar.set_ticklabels([str(tick) for tick in int_ticks])  # Convert to strings for display

    plt.tight_layout()

    # Display the chart in Streamlit
    st.pyplot(fig)

def display_dataset_info(crew_data):
    st.subheader("Dataset Overview")
    st.write(crew_data.head())
    
    st.subheader("Dataset Information")

    buffer = io.StringIO()
    crew_data.info(buf=buffer)  # Write dataset info to the buffer
    info_str = buffer.getvalue()  # Retrieve the string content
    st.text(info_str)
    
    st.subheader("Column Names")
    st.write(crew_data.columns.tolist())

def plot_task_distribution(crew_data):
    st.subheader("Task Distribution")
    task_distribution = crew_data['task_Name'].value_counts()
    fig, ax = plt.subplots(figsize=(12, 8))
    task_distribution.plot(kind='bar', ax=ax)
    ax.set_title('Distribution of Tasks')
    ax.set_xlabel('Task Name')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(task_distribution.index, rotation=45, ha='right')
    st.pyplot(fig)

def optimize_task_allocation(predictions_df):
    model = LpProblem("Task_Reallocation", LpMinimize)
    tasks = predictions_df['Task_ID'].unique()
    gates = predictions_df['Gate_number'].unique()
    shifts = predictions_df['Shift_no'].unique()
    assignments = LpVariable.dicts("assign", ((t, g, s) for t in tasks for g in gates for s in shifts), lowBound=0, cat='Integer')
    
    model += lpSum(assignments[t, g, s] * abs(predictions_df[
        (predictions_df['Task_ID'] == t) &
        (predictions_df['Gate_number'] == g) &
        (predictions_df['Shift_no'] == s)]['Predicted'].values[0])
        for t in tasks for g in gates for s in shifts if predictions_df[
        (predictions_df['Task_ID'] == t) &
        (predictions_df['Gate_number'] == g) &
        (predictions_df['Shift_no'] == s)]['Predicted'].values[0] > 0)
    
    for t in tasks:
        for g in gates:
            for s in shifts:
                predicted_value = predictions_df[
                    (predictions_df['Task_ID'] == t) &
                    (predictions_df['Gate_number'] == g) &
                    (predictions_df['Shift_no'] == s)
                ]['Predicted'].values[0]
                
                if predicted_value > 0:
                    model += assignments[t, g, s] == predicted_value
                else:
                    model += assignments[t, g, s] == 0
    
    model.solve()
    
    results = []
    for t in tasks:
        for g in gates:
            for s in shifts:
                allocated = value(assignments[t, g, s])
                if allocated is not None and allocated > 0:
                    results.append({'Task_ID': t, 'Gate_number': g, 'Shift_no': s, 'Allocated': allocated})
    
    return pd.DataFrame(results)

def plot_heatmap(optimized_allocation):
    st.subheader("Heatmap of Optimized Task Allocation")
    pivot_table = optimized_allocation.pivot_table(values='Allocated', index='Task_ID', columns='Shift_no', aggfunc='first')
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, ax=ax)
    ax.set_title('Optimized Task Allocation by Shift')
    st.pyplot(fig)
def plot_heatmap2(optimized_allocation):
    st.subheader("Heatmap of Optimized Task Allocation")
    pivot_table = optimized_allocation.pivot_table(values='Allocated', index='Task_ID', columns=['Shift_no', 'Gate_number'], aggfunc='first')
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".0f", linewidths=0.5, ax=ax)
    ax.set_title('Optimized Task Allocation by Shift and Gate')
    st.pyplot(fig)

def plot_bar_chart(optimized_allocation):
    st.subheader("Bar Chart of Optimized Task Allocation")

    # Prepare data for the bar chart
    bar_chart_data = optimized_allocation.groupby(['Shift_no', 'Gate_number', 'Task_ID'])['Allocated'].sum().reset_index()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(
        data=bar_chart_data,
        x='Task_ID',
        y='Allocated',
        hue='Shift_no',  # Differentiate bars by shift numbers
        dodge=True,  # Separate bars by 'Shift_no'
        ax=ax
    )

    # Customize the chart
    ax.set_title('Bar Chart of Optimized Task Allocation', fontsize=16)
    ax.set_xlabel('Task ID', fontsize=14)
    ax.set_ylabel('Allocated Value', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate Task_ID labels for readability
    ax.legend(title='Shift Number', fontsize=12)
    
    st.pyplot(fig)
def plot_enhanced_heatmap(optimized_allocation):
    st.subheader("Enhanced Heatmap of Optimized Task Allocation")

    # Pivot the DataFrame to prepare it for heatmap
    heatmap_data = optimized_allocation.pivot_table(
        values='Allocated', 
        index='Task_ID', 
        columns=['Shift_no', 'Gate_number'], 
        aggfunc='first'
    )

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(
        heatmap_data, 
        cmap='coolwarm',  # Fancy color palette
        annot=True,       # Show annotations
        fmt=".0f",        # Format as integers
        linewidths=0.3,   # Add gridlines between cells
        linecolor='black',  # Color for the gridlines
        cbar_kws={'label': 'Crew Allocated'}  # Label for the colorbar
    )

    # Add labels and title
    ax.set_title('Heatmap of Optimized Task Allocation', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Shift and Gate', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Task ID', fontsize=14, fontweight='bold', labelpad=10)

    # Customize tick labels
    plt.xticks(fontsize=10, rotation=45, ha='right')  # Rotate and align x-axis labels
    plt.yticks(fontsize=10, rotation=0)  # Keep y-axis labels horizontal

    # Tight layout for better spacing
    plt.tight_layout()

    # Display the heatmap in Streamlit
    st.pyplot(fig)
def plot_stacked_bar_chart(optimized_allocation):
    st.subheader("Stacked Bar Chart of Task Allocations")

    # Prepare data for stacked bar chart
    stacked_data = optimized_allocation.groupby(['Task_ID', 'Shift_no', 'Gate_number'])['Allocated'].sum().reset_index()

    # Pivot to prepare for stacking
    stacked_pivot = stacked_data.pivot_table(
        values='Allocated',
        index='Task_ID',
        columns=['Shift_no', 'Gate_number'],
        aggfunc='sum'
    ).fillna(0)

    # Create figure with adjusted size to accommodate bottom legend
    fig, ax = plt.subplots(figsize=(15, 10))

    # Plot the stacked bar chart
    stacked_pivot.plot(
        kind='bar',
        stacked=True,
        colormap='viridis',
        linewidth=0.5,
        edgecolor='black',
        ax=ax
    )

    # Add chart details
    ax.set_title('Stacked Bar Chart of Task Allocations', fontsize=16, weight='bold')
    ax.set_xlabel('Task ID', fontsize=14, labelpad=10)
    ax.set_ylabel('Allocated Value', fontsize=14, labelpad=10)
    ax.set_xticklabels(stacked_pivot.index, rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Move legend to bottom and adjust its parameters
    ax.legend(
        title='Shift & Gate',
        fontsize=10,
        title_fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=4  # Adjust number of columns in legend
    )

    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to accommodate bottom legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Increase bottom margin for legend

    # Display the chart in Streamlit
    st.pyplot(fig)

    
def prepare_data(df):
    categorical_cols = ['task_ID', 'Gate_number', 'Floor_No', 'shift_no']
    df[categorical_cols] = df[categorical_cols].astype('category')

    sequences, target, scalers = [], [], {}
    for name, group in df.groupby(categorical_cols, observed=False):
        demand_values = group['crew_demand'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_demand = scaler.fit_transform(demand_values)
        scalers[name] = scaler
        scaled_demand = scaled_demand.flatten()

        seq_length = 10
        for i in range(len(scaled_demand) - seq_length):
            sequences.append((name, scaled_demand[i:i + seq_length]))
            target.append(scaled_demand[i + seq_length])

    X = np.array([seq[1] for seq in sequences])
    y = np.array(target)
    groups = [seq[0] for seq in sequences]
    
    return X, y, groups, scalers

def build_model(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        keras.layers.LSTM(64, activation='tanh', return_sequences=True),
        keras.layers.LSTM(32, activation='tanh'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_evaluate_model(X, y, groups):
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=groups
    )
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.1, verbose=1)
    
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    predictions_scaled = model.predict(X_test)
    
    return model, X_test, y_test, groups_test, predictions_scaled, mae

import streamlit as st
import os
import pandas as pd
import numpy as np

def main():
    st.title("Crew Demand Prediction And Dynamic Crew Reallocation")
    st.subheader("Select Airport terminal")
    option = st.selectbox("Format: City Terminal No",[
        "Bengaluru Terminal 1", "Bengaluru Terminal 2", "Delhi Terminal 1", 
        "Delhi Terminal 2", "Delhi Terminal 3", "Mumbai Terminal 1", 
        "Mumbai Terminal 2", "Hyderabad Terminal 1"
    ])
    
    paths = {
        "Mumbai Terminal 1": "../dataset/MT1/predMT1.csv",
        "Mumbai Terminal 2": "../dataset/MT2/predMT2.csv",
        "Bengaluru Terminal 1": "../dataset/BT1/predBT1.csv",
        "Bengaluru Terminal 2": "../dataset/BT2/predBT2.csv",
        "Delhi Terminal 2": "../dataset/DT2/predDT2.csv",
        "Delhi Terminal 1": "../dataset/DT1/predDT1.csv",
        "Delhi Terminal 3": "../dataset/DT3/predDT3.csv",
        "Hyderabad Terminal 1": "../dataset/HT1/predHT1.csv",
    }
    paths2 = {
        "Mumbai Terminal 1": "../dataset/MT1/crew_dataMT1.csv",
        "Mumbai Terminal 2": "../dataset/MT2/crew_dataMT2.csv",
        "Bengaluru Terminal 1": "../dataset/BT1/crew_dataBT1.csv",
        "Bengaluru Terminal 2": "../dataset/BT2/crew_dataBT2.csv",
        "Delhi Terminal 2": "../dataset/DT2/crew_dataDT2.csv",
        "Delhi Terminal 1": "../dataset/DT1/crew_dataDT1.csv",
        "Delhi Terminal 3": "../dataset/DT3/crew_dataDT3.csv",
        "Hyderabad Terminal 1": "../dataset/HT1/crew_dataHT1.csv",
    }
    st.subheader("Or upload your custom data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X, y, groups, scalers = prepare_data(df)
        st.write("Data Processed Successfully!")
        
        if st.button("Train Model and Predict"):
            model, X_test, y_test, groups_test, predictions_scaled, mae = train_and_evaluate_model(X, y, groups)
            st.write(f"Mean Absolute Error: {mae}")
            
            predictions, groups_used = [], []
            for i, group in enumerate(groups_test):
                scaler = scalers[group]
                prediction_scaled = predictions_scaled[i].reshape(-1, 1)
                prediction = scaler.inverse_transform(prediction_scaled).flatten()
                prediction_adjusted = np.where(prediction < 0, np.ceil(prediction), np.floor(prediction))
                predictions.append(int(prediction_adjusted[0]))
                groups_used.append(group)
            
            predictions_df = pd.DataFrame({
                'Task_ID': [group[0] for group in groups_used],
                'Gate_number': [group[1] for group in groups_used],
                'Floor_No': [group[2] for group in groups_used],
                'Shift_no': [group[3] for group in groups_used],
                'Predicted': predictions
            })
            st.write(predictions_df.head())
            
            csv = predictions_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
    crew_data = st.file_uploader("Upload Crew dataset (CSV)", type=["csv"])

    if "df_predictions" not in st.session_state:
        st.session_state.df_predictions = None
    
    if st.button("Show Predictions"):
        file_path = paths.get(option)
        crew_file_path = paths2.get(option)
        if file_path and os.path.exists(file_path):
            df_predictions = pd.read_csv(file_path)
            df_predictions = df_predictions.drop(columns=["Actual"], errors='ignore')
            df_predictions["Predicted"] = df_predictions["Predicted"].astype(int)
            st.session_state.df_predictions = df_predictions  # Store in session_state
            st.write(df_predictions.head())
            
            csv = df_predictions.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
            
    if st.session_state.df_predictions is not None:
        if st.button("Dynamically Allocate Crew"):
            st.title("Dynamic Crew Allocation and Analysis")
            crew_file_path = paths2.get(option)
            if crew_file_path and os.path.exists(crew_file_path):
                crew_data = pd.read_csv(crew_file_path)
                display_dataset_info(crew_data)
                plot_task_distribution(crew_data)
            else:
                st.error("Crew data file not found!")
    
            optimized_allocation = optimize_task_allocation(st.session_state.df_predictions)
            optimized_allocation.to_csv('optimized_task_allocation.csv', index=False)
    
            st.subheader("Optimized Allocation DataFrame")
            optimized_allocation['Allocated'] = optimized_allocation['Allocated'].astype(int)
            st.write(optimized_allocation)
    
            plot_heatmap(optimized_allocation)
            plot_heatmap2(optimized_allocation)
            plot_bar_chart(optimized_allocation)
            plot_enhanced_heatmap(optimized_allocation)
            plot_stacked_bar_chart(optimized_allocation)
            plot_3d_bar_chart(optimized_allocation)
    
    

if __name__ == "__main__":
    main()
