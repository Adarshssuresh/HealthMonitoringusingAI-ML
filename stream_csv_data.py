import requests
import pandas as pd
import time

# Load CSV file
file_path = "PPG_Dataset.csv"  # Update with the correct path
df = pd.read_csv(file_path)

# Assuming the CSV has 'heart_rate' and 'spo2' columns
for index, row in df.iterrows():
    data = {"heart_rate": row["Heart_Rate"], "spo2": row["SpO2"]}

    # Send request to FastAPI server
    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    # Print the response from the server
    print(f"Input: {data}, Output: {response.json()}")

    # Simulate real-time streaming by adding a delay
    time.sleep(1)  # Adjust delay as needed
