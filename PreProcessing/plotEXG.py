# Define time axis
df_cleaned["Time (s)"] = (df_cleaned["Timestamp"] - df_cleaned["Timestamp"].min()) 

# Plot EEG signals
plt.figure(figsize=(12, 6))
for i, channel in enumerate(["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3"]):
    plt.subplot(4, 1, i + 1)
    plt.plot(df_cleaned["Time (s)"], df_cleaned[channel], label=channel)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.title(channel)
    plt.legend()
    plt.tight_layout()

plt.show()