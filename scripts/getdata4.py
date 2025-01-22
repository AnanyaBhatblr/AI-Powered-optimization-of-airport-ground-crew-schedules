import pandas as pd
import random

# Load the existing CSV file
def add_crew_demand(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drops columns with "Unnamed" in their names
    
    # List of unique combinations for task_ID, Gate_number, and Floor_No
    unique_combinations = df[['task_ID', 'Gate_number', 'Floor_No']].drop_duplicates()

    # Create a list to store the expanded rows
    expanded_rows = []

    # For each unique combination, add 4 shifts
    for _, row in unique_combinations.iterrows():
        for shift in range(1, 5):  # Generate 4 shifts per combination
            new_row = row.copy()
            new_row['shift_no'] = shift# Assign random crew demand
            expanded_rows.append(new_row)

    # Create a new DataFrame with the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Save the modified DataFrame to a new CSV file
    expanded_df.to_csv(output_csv, index=False)

# Specify file paths
input_csv = "../data/task_priority_assignment.csv"  # Replace with your input CSV file path
output_csv = "output_with_crew_demand.csv"  # Output file path

# Add crew demand and save the updated CSV
add_crew_demand(input_csv, output_csv)
print(f"Updated CSV saved as {output_csv}")
