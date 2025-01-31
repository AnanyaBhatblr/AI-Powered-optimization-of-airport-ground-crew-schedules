import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import io
def load_data():
    crew_data = pd.read_csv("../data/crew_data.csv")
    predictions_df = pd.read_csv("../data/predictions_final.csv")
    return crew_data, predictions_df

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

    
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np

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

def main():
    st.title("Crew Data Analysis and Optimization")
    crew_data, predictions_df = load_data()
    
    display_dataset_info(crew_data)
    plot_task_distribution(crew_data)
    
    optimized_allocation = optimize_task_allocation(predictions_df)
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
