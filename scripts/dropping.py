import pandas as pd

# Read the CSV file
df = pd.read_csv('../data/predictions_output.csv')

# Drop the Actual column
df = df.drop('Actual', axis=1)

# Save back to the same file
df.to_csv('../data/predictions_final.csv', index=False)

print("Actual column removed successfully!")