import os
import sys
from google.cloud import storage
import tomllib

def list_gcs_folders():
    if not os.path.exists("secrets.toml"):
        print("secrets.toml not found")
        return

    try:
        with open("secrets.toml", "rb") as f:
            secrets = tomllib.load(f)
    except Exception as e:
        print(f"Error loading secrets: {e}")
        return

    if "gcp_service_account" not in secrets:
        print("No GCP credentials in secrets.toml")
        return

    try:
        client = storage.Client.from_service_account_info(secrets["gcp_service_account"])
        bucket_name = secrets["general"]["bucket_name"]
        bucket = client.bucket(bucket_name)

        print(f"Listing top-level folders in bucket: {bucket_name}")
        
        # List blobs with delimiter to simulate folders
        blobs = bucket.list_blobs(delimiter="/")
        list(blobs) # trigger fetching to populate prefixes

        print("\nTop-level folders (prefixes):")
        for prefix in blobs.prefixes:
            print(f" - {prefix}")

    except Exception as e:
        print(f"GCS Error: {e}")

if __name__ == "__main__":
    list_gcs_folders()
