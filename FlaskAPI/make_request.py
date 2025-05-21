import requests
from data_input import data_in

URL = 'http://127.0.0.1:5000/predict'
headers = {"Content-Type": "application/json"}
data = {"input": data_in}

r = requests.post(URL, headers=headers, json=data)

print("Status Code:", r.status_code)
print("Response Text:", r.text)

try:
    print("JSON Response:", r.json())
except Exception as e:
    print("Failed to decode JSON:", e)