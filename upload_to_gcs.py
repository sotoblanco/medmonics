import sys
import os
import tomllib
from pathlib import Path
import json

# Add current directory to path
sys.path.append(os.getcwd())

from medmonics.storage import GCSBackend

def migrate():
    print("🚀 Starting migration to GCS...")
    
    # 1. Load Secrets
    if not os.path.exists("secrets.toml"):
        print("❌ secrets.toml not found!")
        return

    with open("secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
        
    # 2. Initialize Backend
    try:
        backend = GCSBackend(
            bucket_name=secrets["general"]["bucket_name"],
            service_account_info=secrets["gcp_service_account"]
        )
        print("✅ Connected to GCS")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return

    # 3. Walk generations directory
    root_dir = Path("generations")
    if not root_dir.exists():
        print("No generations directory found.")
        return

    count = 0
    # Walk through local directory
    for item in root_dir.iterdir():
        if not item.is_dir() or item.name == "batch_runs" or item.name == "batch_import":
            continue
            
        # Check if item is a specialty folder (contains subfolders) OR a generation folder (contains data.json)
        is_generation = (item / "data.json").exists()
        
        if is_generation:
            # It's a legacy/flat generation folder
            # We'll put it in "general" or try to infer specialty from metadata?
            # For simplicity, put in "general"
            print(f"📦 Found flat generation: {item.name}")
            specialty_name = "general"
            gen_dirs = [item]
        else:
            # It's likely a specialty folder
            print(f"📂 Processing Specialty: {item.name}")
            specialty_name = item.name
            gen_dirs = [d for d in item.iterdir() if d.is_dir()]

        for gen_dir in gen_dirs:
            data_path = gen_dir / "data.json"
            image_path = gen_dir / "image.png"
            
            if data_path.exists():
                try:
                    # Remote path: specialty/timestamp_topic/filename
                    # If it was a flat generation, use its name
                    if is_generation:
                         # flat folder "2026..." -> "general/2026..."
                         remote_folder = f"{specialty_name}/{gen_dir.name}"
                    else:
                         # nested "cardiology/2026..." -> "cardiology/2026..."
                         remote_folder = f"{specialty_name}/{gen_dir.name}"
                         
                    print(f"   ⬆️ Uploading {gen_dir.name} to {remote_folder}...")
                    
                    # Upload Image
                    if image_path.exists():
                        blob_img = backend.bucket.blob(f"{remote_folder}/image.png")
                        if not blob_img.exists():
                            blob_img.upload_from_filename(str(image_path), content_type="image/png")
                    
                    # Upload Data
                    blob_data = backend.bucket.blob(f"{remote_folder}/data.json")
                    if not blob_data.exists():
                        blob_data.upload_from_filename(str(data_path), content_type="application/json")
                    else:
                        print(f"      - Skipping (already exists)")
                    
                    count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error uploading {gen_dir.name}: {e}")
            
    print(f"\n✨ Migration Complete! Uploaded {count} generations.")

if __name__ == "__main__":
    migrate()
