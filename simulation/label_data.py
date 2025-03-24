import pandas as pd
import json

# Load the collected data
data = pd.read_csv("resource_exhaustion_data.csv", parse_dates=['timestamp'], index_col='timestamp')

# Load stress events
with open("stress_events.json", "r") as f:
    stress_events = pd.DataFrame(json.load(f))
stress_events['timestamp'] = pd.to_datetime(stress_events['timestamp'])

# Define exhaustion thresholds
CPU_THRESHOLD = 0.45  # 90% of 500m
MEMORY_THRESHOLD = 460 * 1024 * 1024  # 460Mi in bytes

# Function to label exhaustion
def label_exhaustion(row):
    # Check CPU and memory for each pod
    for col in row.index:
        if 'container_cpu_usage_seconds_total' in col:
            if row[col] > CPU_THRESHOLD:
                return 1
        if 'container_memory_working_set_bytes' in col:
            if row[col] > MEMORY_THRESHOLD:
                return 1
    return 0

# Apply exhaustion label
data['exhaustion'] = data.apply(label_exhaustion, axis=1)

# Create future exhaustion label (15 seconds ahead)
LOOKAHEAD_SECONDS = 15
data['future_exhaustion'] = data['exhaustion'].shift(-int(LOOKAHEAD_SECONDS / 15))  # 15s / 15s interval = 1 row

# Add stress event information
data['cpu_load'] = 0
data['memory_size'] = 0
for _, event in stress_events.iterrows():
    event_time = event['timestamp']
    # Find the closest timestamp in the data
    closest_time = data.index[data.index.get_loc(event_time, method='nearest')]
    # Set stress values for the next 5 minutes (300 seconds)
    end_time = closest_time + pd.Timedelta(seconds=300)
    mask = (data.index >= closest_time) & (data.index <= end_time)
    data.loc[mask, 'cpu_load'] = event['cpu_load']
    data.loc[mask, 'memory_size'] = event['memory_size'].replace('MB', '').astype(int)

# Drop rows with NaN (due to shift)
data = data.dropna()

# Save the labeled dataset
data.to_csv("labeled_resource_exhaustion_data.csv")
print("Labeled dataset saved to labeled_resource_exhaustion_data.csv")