import boto3
import os
from botocore import UNSIGNED
from botocore.client import Config

# S3 details
BUCKET = "noaa-gestofs-pds"
PREFIX = "_post_processing/_metrics/"
LOCAL_ROOT = "metrics_data"

# S3 client, unsigned (public bucket)
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# Paginate through all objects with the given prefix
paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=BUCKET, Prefix=PREFIX)

for page in pages:
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.endswith('/') or not key.endswith('.csv'):
            continue  # skip folders and non-csvs
        # Split key: e.g. _post_processing/_metrics/20240730/8720218_estofs.csv
        parts = key.split('/')
        if len(parts) < 4:
            continue  # should always be at least 4 parts
        date_folder = parts[2]
        filename = parts[3]
        local_dir = os.path.join(LOCAL_ROOT, date_folder)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {key} -> {local_path}")
            s3.download_file(BUCKET, key, local_path)
        else:
            print(f"Already exists: {local_path}")

print("CSV download complete.")
