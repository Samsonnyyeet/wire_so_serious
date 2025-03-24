import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os

# Prometheus API endpoint
PROMETHEUS_URL = "http://localhost:9090/api/v1/query_range"

# Metrics to collect
METRICS = [
    'kube_pod_status_phase{phase="Failed"}',
    'kube_pod_container_status_restarts_total{namespace="default", pod=~"nginx-deployment-.*"}',
    'rate(container_network_receive_packets_dropped_total{namespace="default", pod=~"nginx-deployment-.*"}[5m])',
    'rate(container_network_receive_bytes_total{namespace="default", pod=~"nginx-deployment-.*"}[5m])',
    'up{job="nginx-exporter"}',
    'nginx_requests_total',
    'rate(container_cpu_usage_seconds_total{namespace="default", pod=~"nginx-deployment-.*"}[5m])',
    'container_memory_working_set_bytes{namespace="default", pod=~"nginx-deployment-.*"}'
]

# Data collection parameters
INTERVAL = 15  # Collect data every 15 seconds
DURATION = 3600  # Collect data for 1 hour (3600 seconds)
STEP = "15s"  # Prometheus query step

# File to log stress events
STRESS_LOG_FILE = "stress_events.json"

def collect_metrics(start_time, end_time):
    data = {}
    for metric in METRICS:
        print(f"Collecting {metric}...")
        params = {
            "query": metric,
            "start": start_time.isoformat() + "Z",
            "end": end_time.isoformat() + "Z",
            "step": STEP
        }
        response = requests.get(PROMETHEUS_URL, params=params)
        if response.status_code != 200:
            print(f"Error collecting {metric}: {response.status_code}")
            continue
        data[metric] = response.json()
        # Save raw JSON data
        metric_name = metric.split("{")[0]  # Simplify metric name for filename
        with open(f"{metric_name}_raw.json", "w") as f:
            json.dump(data[metric], f)
    return data

def process_data(data):
    all_dfs = []
    for metric, result in data.items():
        if result['status'] != 'success' or result['data']['resultType'] != 'matrix':
            print(f"Skipping {metric}: Invalid data")
            continue

        metric_data = []
        for res in result['data']['result']:
            pod = res['metric'].get('pod', 'unknown')
            for timestamp, value in res['values']:
                metric_data.append({
                    'timestamp': pd.to_datetime(timestamp, unit='s'),
                    'pod': pod,
                    'metric': metric,
                    'value': float(value)
                })

        df = pd.DataFrame(metric_data)
        df_pivot = df.pivot(index='timestamp', columns=['pod', 'metric'], values='value')
        all_dfs.append(df_pivot)

    # Combine all metrics into a single DataFrame
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=1)
        return combined_df
    return None

def log_stress_event(cpu_load, memory_size):
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu_load": cpu_load,
        "memory_size": memory_size
    }
    if os.path.exists(STRESS_LOG_FILE):
        with open(STRESS_LOG_FILE, "r") as f:
            events = json.load(f)
    else:
        events = []
    events.append(event)
    with open(STRESS_LOG_FILE, "w") as f:
        json.dump(events, f)

def main():
    print("Starting data collection...")
    start_time = datetime.utcnow()
    end_time = start_time + timedelta(seconds=DURATION)

    # Collect data in real-time
    current_time = start_time
    all_data = []

    while current_time < end_time:
        next_time = current_time + timedelta(seconds=INTERVAL)
        data = collect_metrics(current_time, next_time)
        df = process_data(data)
        if df is not None:
            all_data.append(df)
        current_time = next_time
        time.sleep(INTERVAL)

    # Combine all collected data
    if all_data:
        final_df = pd.concat(all_data).sort_index()
        final_df.to_csv("resource_exhaustion_data.csv")
        print("Data saved to resource_exhaustion_data.csv")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()