import pandas as pd

# Load the data
def loadData(filepath= "CombinedTemp.txt"):

    with open(filepath, "r") as f:
        lines = f.readlines()

    start_idx = next(i for i, line in enumerate(lines) if not line.startswith('%')) # AI generated

    dataframe = pd.read_csv(filepath, skiprows=start_idx) # Put data into pandas dataframe

    dataframe.rename(columns={' EXG Channel 0': 'EXG Channel 0', ' EXG Channel 1': 'EXG Channel 1',' EXG Channel 2': 'EXG Channel 2',' EXG Channel 3': 'EXG Channel 3',' Timestamp': 'Timestamp', ' Marker Channel': 'Marker Channel',' Timestamp (Formatted)': 'Timestamp (Formatted)',}, inplace=True) # Rename columns as they have spaces in them

    # Drop unneeded columns
    columns_to_drop = [" Other", " Other.1", " Other.2", " Other.3", " Other.4", " Accel Channel 0", " Accel Channel 1", " Accel Channel 2"]
    df_cleaned = dataframe.drop(columns=[col for col in columns_to_drop])

    # Convert data types (keeping Timestamp (Formatted) and Marker Channel)
    df_cleaned = df_cleaned.astype({
        "Sample Index": int,
        "EXG Channel 0": float,
        "EXG Channel 1": float,
        "EXG Channel 2": float,
        "EXG Channel 3": float,
        "Timestamp": float,
        "Marker Channel": float
    })
    return df_cleaned
