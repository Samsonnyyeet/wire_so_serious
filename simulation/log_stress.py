import json
from datetime import datetime
import sys
import os

def log_stress_event(cpu_load, memory_size):
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "cpu_load": int(cpu_load),
        "memory_size": memory_size
    }
    filename = "stress_events.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            events = json.load(f)
    else:
        events = []
    events.append(event)
    with open(filename, "w") as f:
        json.dump(events, f)

if __name__ == "__main__":
    cpu_load = sys.argv[1]
    memory_size = sys.argv[2]
    log_stress_event(cpu_load, memory_size)