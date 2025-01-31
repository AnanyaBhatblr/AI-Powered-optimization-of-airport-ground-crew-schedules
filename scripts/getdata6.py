import pandas as pd

def add_personnel_to_dataset(dataset1_csv, dataset2_csv, output_csv):
    # Read the datasets
    dataset1 = pd.read_csv(dataset1_csv)
    dataset2 = pd.read_csv(dataset2_csv)

    # Remove any unnamed columns
    dataset1 = dataset1.loc[:, ~dataset1.columns.str.contains('^Unnamed')]
    dataset2 = dataset2.loc[:, ~dataset2.columns.str.contains('^Unnamed')]

    # Merge the datasets on task_ID, Gate_number, and Floor_No
    merged_df = pd.merge(dataset2, dataset1[['task_ID', 'Gate_number', 'Floor_No', 'Number_of_Personnel']],
                         on=['task_ID', 'Gate_number', 'Floor_No'],
                         how='left')

    # Save the updated dataset to a new CSV file
    merged_df.to_csv(output_csv, index=False)

# Specify file paths
dataset1_csv = "../dataset/BT1/task_personnel_summaryBT1.csv"  # Replace with the path to Dataset 1
dataset2_csv = "../dataset/BT1/output_with_datesBT1.csv"  # Replace with the path to Dataset 2
output_csv = "../dataset/BT1/outputBT1.csv"  # Output file path

# Add Number_of_Personnel to Dataset 2 and save
add_personnel_to_dataset(dataset1_csv, dataset2_csv, output_csv)
print(f"Updated Dataset 2 saved as {output_csv}")
