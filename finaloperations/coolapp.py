import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

# Load data
crew_data = pd.read_csv("../dataset/DT1/crew_dataDT1.csv")
predictions_df = pd.read_csv("../dataset/DT1/predDT1.csv")

# Define optimization function
def optimize_task_allocation(predictions_df):
    model = LpProblem("Task_Reallocation", LpMinimize)
    tasks = predictions_df['Task_ID'].unique()
    gates = predictions_df['Gate_number'].unique()
    shifts = predictions_df['Shift_no'].unique()
    
    assignments = LpVariable.dicts(
        "assign",
        ((t, g, s) for t in tasks for g in gates for s in shifts),
        lowBound=0,  
        cat='Integer'
    )
    
    model += lpSum(
        assignments[t, g, s] * abs(predictions_df[
            (predictions_df['Task_ID'] == t) &
            (predictions_df['Gate_number'] == g) &
            (predictions_df['Shift_no'] == s)]['Predicted'].values[0])
        for t in tasks for g in gates for s in shifts
        if predictions_df[
            (predictions_df['Task_ID'] == t) &
            (predictions_df['Gate_number'] == g) &
            (predictions_df['Shift_no'] == s)]['Predicted'].values[0] > 0
    )
    
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
                    results.append({
                        'Task_ID': t,
                        'Gate_number': g,
                        'Shift_no': s,
                        'Allocated': allocated
                    })
    
    return pd.DataFrame(results)

optimized_allocation = optimize_task_allocation(predictions_df)

# Streamlit app
st.title("Crew Task Allocation Dashboard")
st.write("### Crew Data Preview")
st.dataframe(crew_data.head())

# Task Distribution Analysis
task_distribution = crew_data['task_Name'].value_counts()
st.write("### Task Distribution")
st.bar_chart(task_distribution)

# Shift Pattern Analysis
shift_distribution = crew_data['Shift_No'].value_counts()
st.write("### Shift Distribution")
st.bar_chart(shift_distribution)

# Optimized Task Allocation Heatmap
st.write("### Optimized Task Allocation Heatmap")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
heatmap_data = optimized_allocation.pivot_table(
    values='Allocated', 
    index='Task_ID', 
    columns=['Shift_no', 'Gate_number'],
    aggfunc='first'
)
plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".0f", linewidths=0.5)
plt.title('Heatmap of Optimized Task Allocation')
plt.xlabel('Shift and Gate')
plt.ylabel('Task ID')
plt.tight_layout()
st.pyplot(plt)
