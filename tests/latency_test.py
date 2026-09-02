import time
import statistics
import requests

API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.20
}

NUM_REQUESTS = 20

latencies = []

print("\n======================================")
print("Starting API Latency Test")
print("======================================")

for i in range(NUM_REQUESTS):

    start = time.perf_counter()

    response = requests.post(API_URL, json=payload)

    end = time.perf_counter()

    latency_ms = (end - start) * 1000

    latencies.append(latency_ms)

    print(
        f"Request {i+1:02d} | "
        f"Status: {response.status_code} | "
        f"Latency: {latency_ms:.2f} ms"
    )

print("\n======================================")
print("LATENCY REPORT")
print("======================================")

print(f"Requests Tested : {NUM_REQUESTS}")

print(f"Average Latency : {statistics.mean(latencies):.2f} ms")

print(f"Minimum Latency : {min(latencies):.2f} ms")

print(f"Maximum Latency : {max(latencies):.2f} ms")

latencies.sort()

p95 = latencies[int(NUM_REQUESTS * 0.95) - 1]

print(f"P95 Latency     : {p95:.2f} ms")

print("======================================")