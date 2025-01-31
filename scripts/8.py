import pandas as pd
import numpy as np
import random

def add_crew_demand_combined_patterns(input_csv, output_csv):
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file '{input_csv}' not found.")
        return

    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    num_rows = len(df)
    crew_demand = []

    # Define multiple patterns
    patterns = [
        [0, 1, 2, 1, 0, -1, -2, -1],  # Pattern 1 (period 8)
        [0, 2, 0, -2],              # Pattern 2 (period 4)
        [0, 1, 0, -1, 0]           # Pattern 3 (period 5)
    ]
    pattern_lengths = [len(p) for p in patterns]

    for i in range(num_rows):
        demand = 0
        for j, pattern in enumerate(patterns):
            pattern_index = i % pattern_lengths[j]
            demand += pattern[pattern_index]

        demand += random.randint(-1, 1)  # Add some noise
        demand = max(-2, min(5, demand))
        crew_demand.append(demand)

    df['crew_demand'] = crew_demand
    try:
        df.to_csv(output_csv, index=False)
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return

    print(f"Dataset with combined patterns saved as {output_csv}")


# Example usage:
input_csv = "../dataset/BT1/outputBT1.csv"  # Replace with your input CSV file path
output_csv = "../dataset/BT1/output2BT1.csv"

# Create a sample input CSV if it doesn't exist2
try:
    test_df = pd.read_csv(input_csv)
except FileNotFoundError:
    print(f"Creating a sample {input_csv} file...")
    data = {'task_ID': [1, 2, 1, 2, 1, 2]*20, 'gate_no': ['A1', 'B2', 'A1', 'B2','A1', 'B2']*20, 'floor_no': [1, 2, 1, 2, 1, 2]*20, 'shift_no': [1, 2, 1, 2, 1, 2]*20}
    test_df = pd.DataFrame(data)
    test_df.to_csv(input_csv, index=False)
    print(f"Sample {input_csv} file created.")

add_crew_demand_combined_patterns(input_csv, output_csv)