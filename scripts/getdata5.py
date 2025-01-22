import pandas as pd
import random
from datetime import datetime, timedelta

def add_dates_to_data(input_csv, output_csv):
    # Read the input CSV file
    df = pd.read_csv(input_csv)
    
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drops columns with "Unnamed" in their names
    
    # Create a list of 4 random dates for each month between January 1st, 2023 and December 31st, 2023
    dates_per_month = []
    
    for month in range(1, 13):  # For each month from January (1) to December (12)
        # Get the first and last day of the month
        start_date = datetime(2023, month, 1)
        end_date = datetime(2023, month + 1, 1) - timedelta(days=1) if month < 12 else datetime(2023, 12, 31)
        
        # Generate 4 random dates within this month
        random_dates = []
        while len(random_dates) < 4:
            random_day = random.randint(start_date.day, end_date.day)
            random_date = datetime(2023, month, random_day)
            if random_date not in random_dates:
                random_dates.append(random_date)
        
        dates_per_month.extend(random_dates)  # Add these 4 dates to the list
    
    # Create an empty list to store the expanded rows with dates
    expanded_rows = []
    
    # For each row in the original dataframe, assign the 4 dates
    for _, row in df.iterrows():
        for date in dates_per_month:
            new_row = row.copy()
            new_row['date'] = date  # Assign the date to the new row
            expanded_rows.append(new_row)
    
    # Create a new DataFrame with the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Save the modified DataFrame to a new CSV file
    expanded_df.to_csv(output_csv, index=False)

# Specify file paths
input_csv = "output_with_crew_demand.csv"  # Replace with your input CSV file path
output_csv = "output_with_dates1.csv"  # Output file path

# Add dates to the data and save the updated CSV
add_dates_to_data(input_csv, output_csv)
print(f"Updated CSV saved as {output_csv}")
