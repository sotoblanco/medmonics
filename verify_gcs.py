import sys
import os
import toml

# Add current directory to path so we can import medmonics package
sys.path.append(os.getcwd())

from medmonics.storage import GCSBackend

try:
    print("Loading secrets from .streamlit/secrets.toml...")
    secrets = toml.load(".streamlit/secrets.toml")
    
    print("Initializing GCS Backend...")
    backend = GCSBackend(
        bucket_name=secrets["general"]["bucket_name"],
        service_account_info=secrets["gcp_service_account"]
    )
    
    print("Listing generations to verify connection...")
    generations = backend.list_generations()
    print(f"✅ Connection Successful! Found {len(generations)} existing generations.")
    
except Exception as e:
    print(f"❌ Verification Failed: {e}")
    import traceback
    traceback.print_exc()
