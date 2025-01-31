import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
st.title("Crew Data Analysis")
crew_data = pd.read_csv("../data/crew_data.csv")

# Display dataset info
st.subheader("Dataset Overview")
st.write(crew_data.head())

st.subheader("Dataset Information")
st.text(crew_data.info())

st.subheader("Column Names")
st.write(crew_data.columns.tolist())

# Task Distribution Analysis
st.subheader("Task Distribution")
task_distribution = crew_data['task_Name'].value_counts()

fig, ax = plt.subplots(figsize=(12, 8))
task_distribution.plot(kind='bar', ax=ax)
ax.set_title('Distribution of Tasks')
ax.set_xlabel('Task Name')
ax.set_ylabel('Frequency')
ax.set_xticklabels(task_distribution.index, rotation=45, ha='right')
st.pyplot(fig)

# Optimization Function
def optimize_task_allocation(predictions_df):
    model = LpProblem("Task_Reallocation", LpMinimize)
    tasks, gates, shifts = (predictions_df['Task_ID'].unique(), 
                            predictions_df['Gate_number'].unique(), 
                            predictions_df['Shift_no'].unique())
    assignments = LpVariable.dicts("assign", ((t, g, s) for t in tasks for g in gates for s in shifts), lowBound=0, cat='Integer')
    model += lpSum(assignments[t, g, s] * abs(predictions_df.loc[(predictions_df['Task_ID'] == t) &
                                                                 (predictions_df['Gate_number'] == g) &
                                                                 (predictions_df['Shift_no'] == s), 'Predicted'].values[0])
                   for t in tasks for g in gates for s in shifts)
    for t in tasks:
        for g in gates:
            for s in shifts:
                predicted_value = predictions_df.loc[(predictions_df['Task_ID'] == t) & (predictions_df['Gate_number'] == g) & (predictions_df['Shift_no'] == s), 'Predicted'].values[0]
                model += assignments[t, g, s] == predicted_value if predicted_value > 0 else assignments[t, g, s] == 0
    model.solve()
    return pd.DataFrame([{ 'Task_ID': t, 'Gate_number': g, 'Shift_no': s, 'Allocated': value(assignments[t, g, s]) }
                         for t in tasks for g in gates for s in shifts if value(assignments[t, g, s]) > 0])

# Load and process data
predictions_df = pd.read_csv('../data/predictions_final.csv')
optimized_allocation = optimize_task_allocation(predictions_df)
optimized_allocation.to_csv('optimized_task_allocation.csv', index=False)

st.subheader("Optimized Task Allocation by Shift")
pivot_table = optimized_allocation.pivot_table(values='Allocated', index='Task_ID', columns='Shift_no', aggfunc='first')
st.write(pivot_table)

# Heatmap
st.subheader("Heatmap of Optimized Task Allocation")
fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, ax=ax)
ax.set_title('Optimized Task Allocation by Shift')
st.pyplot(fig)

# Stacked Bar Chart
stacked_data = optimized_allocation.groupby(['Task_ID', 'Shift_no', 'Gate_number'])['Allocated'].sum().reset_index()
stacked_pivot = stacked_data.pivot_table(values='Allocated', index='Task_ID', columns=['Shift_no', 'Gate_number'], aggfunc='sum').fillna(0)
fig, ax = plt.subplots(figsize=(15, 10))
stacked_pivot.plot(kind='bar', stacked=True, colormap='viridis', linewidth=0.5, edgecolor='black', ax=ax)
ax.set_title('Stacked Bar Chart of Task Allocations')
ax.set_xlabel('Task ID')
ax.set_ylabel('Allocated Value')
ax.legend(title='Shift & Gate', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
st.pyplot(fig)

# 3D Bar Plot
optimized_allocation['Task_ID_numeric'] = pd.factorize(optimized_allocation['Task_ID'])[0]
x, y, z, allocated_values = (optimized_allocation['Shift_no'], optimized_allocation['Gate_number'], optimized_allocation['Task_ID_numeric'], optimized_allocation['Allocated'])
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')
norm = plt.Normalize(allocated_values.min(), allocated_values.max())
colors = plt.cm.YlOrRd(norm(allocated_values))
ax.bar3d(x, y, z, 0.5, 0.5, allocated_values, color=colors)
ax.set_xlabel('Shift Number')
ax.set_ylabel('Gate Number')
ax.set_zlabel('Task ID (Numeric)')
ax.set_title('3D Bar Plot of Optimized Task Allocation')
st.pyplot(fig)
