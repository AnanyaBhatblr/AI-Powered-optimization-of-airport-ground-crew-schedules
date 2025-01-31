import pandas as pd

# Load the dataset
input_file = "../dataset/BT1/crew_dataBT1.csv"
data = pd.read_csv(input_file)

# Define criticality levels based on task_ID
criticality_mapping = {
    "T-001": "High",
    "T-002": "Medium",
    "T-003": "Medium",
    "T-004": "Low",
    "T-005": "Medium",
    "T-006": "High",
    "T-007": "Medium",
    "T-008": "Low",
    "T-009": "Medium",
    "T-010": "High",
    "T-011": "Low",
    "T-012": "Low",
    "T-013": "Medium",
    "T-014": "Low",
    "T-015": "Low",
    "T-016": "High",
    "T-017": "Medium",
    "T-018": "High",
    "T-019": "Low",
}

# Map criticality levels to the dataset
data['Criticality'] = data['task_ID'].map(criticality_mapping)

# Replace criticality strings with numeric values
criticality_numeric_mapping = {"High": 3, "Medium": 2, "Low": 1}
data['Criticality'] = data['Criticality'].map(criticality_numeric_mapping)

# Group by task_ID, Floor_No, and Gate_number, and count personnel
summary = (
    data.groupby(['task_ID', 'Floor_No', 'Gate_number', 'Criticality'])
    .size()  # Count the number of personnel for each group
    .reset_index(name='Number_of_Personnel')
)

# Save the aggregated data to a new CSV
output_file = "../dataset/BT1/task_personnel_summaryBT1.csv"
summary.to_csv(output_file, index=False)

print(f"Aggregated dataset with numeric criticality saved to {output_file}")
