import tensorflow as tf
import requests
import subprocess
from pprint import pprint
from utils_test import *
import yaml

with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Get the access token using gcloud
access_token = (
    subprocess.check_output("gcloud auth print-access-token", shell=True)
    .decode("utf-8")
    .strip()
)

ENDPOINT_ID="6608526677817425920" 
GOOGLE_CLOUD_REGION = config["region"]
GOOGLE_CLOUD_PROJECT = config["project"]  
SCHEMA_DIR = "../data/schema.pbtxt"

instances = {
            "trip_month": [2],  
            "trip_day": [7],    
            "trip_day_of_week": [6],  
            "trip_hour": [23],   
            "trip_seconds": 2262, 
            "trip_miles": 11.66,  
            "payment_type": ["Credit Card"],  
            "euclidean": 1702.69,  
            "total_arrests": 3108,  
            "fare": 32.75,   
            "tips": 6.85,   
            "pickup_community_area": [33.0],  
            "dropoff_community_area": [2.0]  
        }

url = f"https://us-west2-aiplatform.googleapis.com/v1/projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/endpoints/{ENDPOINT_ID}:predict"

schema = _read_schema(SCHEMA_DIR)
input_data = {
    "instances": [
        {
            "b64": tf.io.encode_base64(_make_proto_coder(schema).encode(instances)).numpy().decode('utf-8')
        }
    ]
}

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}

# Send the POST request
response = requests.post(url, headers=headers, json=input_data)

# Print the response
pprint(response.json())
