import pandas as pd
import random
from datetime import datetime, timedelta

# Load the existing CSV file
def add_crew_demand(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drops columns with "Unnamed" in their names

    # Generate random values for new columns
    num_rows = len(df)
    df['crew_demand'] = [random.randint(0, 10) for _ in range(num_rows)]  # Random extra crew needed
    df['shift_no'] = [random.randint(1, 4) for _ in range(num_rows)]  # Random shift number (e.g., 1, 2, 3, 4)

    start_date = datetime(2023, 1, 1)  # Start of the date range
    end_date = datetime(2024, 12, 31)  # End of the date range
    df['date'] = [start_date + timedelta(days=random.randint(0, (end_date - start_date).days)) for _ in range(num_rows)]


    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Specify file paths
input_csv = "../data/task_priority_assignment.csv"  # Replace with your input CSV file path
output_csv = "output_with_crew_demand.csv"  # Output file path

# Add crew demand and save the updated CSV
add_crew_demand(input_csv, output_csv)
print(f"Updated CSV saved as {output_csv}")
