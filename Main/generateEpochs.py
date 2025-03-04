
def generate_event_epochs(df, window=(-0.5, 1.0), fs=200):
    epochs = {}
    samples_before = int(abs(window[0]) * fs)  # Number of samples before event
    samples_after = int(window[1] * fs)        # Number of samples after event
    total_samples = samples_before + samples_after

    # Find event timestamps and corresponding marker values
    events = df[df["Marker Channel"].isin([2,3,4,5,6,7,8])][["Timestamp", "Marker Channel"]].values  # Get time + marker value

    print(f"Number of events: {len(events)}")

    for event_time, marker_value in events:
        # Define the epoch time range
        start_time = event_time + window[0]
        end_time = event_time + window[1]

        # Extract data within this time window
        epoch_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()
        
        # Ensure consistent(ish) length
        if len(epoch_df) >= total_samples * 0.9825:
            # Add marker column to indicate event type
            epoch_df["Event Type"] = marker_value  
            epochs[(event_time, marker_value)] = epoch_df
    return epochs

def generate_idle_epochs(df, window=(-0.5, 1.0), fs=200):
    idle_epochs = {}
    samples_before = int(abs(window[0]) * fs)
    samples_after = int(window[1] * fs)
    total_samples = samples_before + samples_after

    # Identify timestamps without events
    event_times = set(df[df["Marker Channel"] != 0]["Timestamp"].values)
    idle_df = df[~df["Timestamp"].isin(event_times)]  # Select non-event rows
    timestamps = idle_df["Timestamp"].values

    i = 0
    while i < len(timestamps) - total_samples:
        start_time = timestamps[i]
        end_time = start_time + (total_samples / fs)

        # Extract potential idle epoch
        epoch_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()

        if len(epoch_df) >= total_samples * 0.9825:  # Ensure sufficient samples
            epoch_df["Event Type"] = -1  # Label as idle
            idle_epochs[(start_time, -1)] = epoch_df
            i += total_samples  # Move forward to avoid overlap
        else:
            i += 1  # Shift forward if the window is too short

    return idle_epochs
