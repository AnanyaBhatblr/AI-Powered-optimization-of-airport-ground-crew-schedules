import pandas as pd
import random

def add_crew_demand(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drops columns with "Unnamed" in their names
    
    # Add a new column 'crew_demand' with random values from -2 to 5
    df['crew_demand'] = [random.randint(-2, 5) for _ in range(len(df))]
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

# Specify file paths
input_csv = "output.csv"  # Replace with your input CSV file path
output_csv = "output1.csv"  # Output file path

# Add the crew demand column and save the updated CSV
add_crew_demand(input_csv, output_csv)
print(f"Updated dataset saved as {output_csv}")
