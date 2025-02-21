import pandas as pd
import argparse

# Set up argument parser to accept the filename as a command-line argument
parser = argparse.ArgumentParser(description='Filter rows in a CSV file based on marker channel.')
parser.add_argument('file_path', type=str, help='Path to the CSV file to process')

# Parse the command-line arguments
args = parser.parse_args()

# Set pandas to display all rows
pd.set_option('display.max_rows', None)

# Load the data
file_path = args.file_path  # Get file path from command-line argument
df = pd.read_csv(file_path, comment='%', delimiter=',')  # Ignores lines starting with '%'

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Check available column names
print("Columns in dataset:", df.columns)

# Filter rows where Marker Channel is not 0 (ensure correct column name)
marker_col = "Marker Channel"  # Update this if the column name is different
if marker_col in df.columns:
    filtered_df = df[df[marker_col] != 0]
    print(filtered_df)
    filtered_df.to_csv("filtered_marked_rows.txt", index=False, sep='\t')
else:
    print(f"Error: Column '{marker_col}' not found in dataset.")
