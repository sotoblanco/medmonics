import os
import sys
from pathlib import Path
import json
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from medmonics.storage import LocalStorage, GCSBackend
    from medmonics.pipeline import MnemonicResponse, QuizList, BboxAnalysisResponse
    import tomllib
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Load variables
load_dotenv()

def get_storage():
    secrets = {}
    
    # Mocking st.secrets not needed if we rely on secrets.toml or env vars for this script
    if os.path.exists("secrets.toml"):
        try:
            with open("secrets.toml", "rb") as f:
                secrets = tomllib.load(f)
            print("Loaded secrets.toml")
        except Exception as e:
            print(f"Failed to load secrets.toml: {e}")

    # Initialize GCS if secrets found
    if "gcp_service_account" in secrets:
        try:
            print(f"Initializing GCS Backend with bucket: {secrets['general']['bucket_name']}")
            return GCSBackend(
                bucket_name=secrets["general"]["bucket_name"],
                service_account_info=secrets["gcp_service_account"]
            )
        except Exception as e:
            print(f"Failed to initialize GCS Backend: {e}")
            return LocalStorage()
            
    print("Falling back to LocalStorage")
    return LocalStorage()

def main():
    storage = get_storage()
    print(f"Storage Backend: {type(storage).__name__}")
    
    print("\n--- Listing Generations ---")
    generations = storage.list_generations()
    print(f"Found {len(generations)} generations.")
    
    if not generations:
        print("No generations found. Check storage bucket/folder.")
        return

    print("\n--- Attempting to Load Generations ---")
    success_count = 0
    fail_count = 0
    
    for gen in generations:
        identifier = gen['identifier']
        name = gen.get('name', 'Unknown')
        print(f"Loading '{name}' (ID: {identifier})...")
        
        try:
            data = storage.load_generation(identifier)
            # data is tuple: (m_data, b_data, q_data, i_bytes, meta)
            m_data, b_data, q_data, i_bytes, meta = data
            
            # Check typing (simulate app logic)
            if isinstance(q_data, dict):
                 q_data = QuizList(**q_data)
                 
            quiz_count = len(q_data.quizzes) if q_data and q_data.quizzes else 0
            print(f"  -> Success! Found {quiz_count} quizzes.")
            success_count += 1
            
        except Exception as e:
            print(f"  -> FAILED: {e}")
            traceback.print_exc()
            fail_count += 1
            
    print(f"\nSummary: {success_count} loaded successfully, {fail_count} failed.")

if __name__ == "__main__":
    main()
