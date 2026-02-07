import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import Optional, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types
from medmonics import prompts

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "models/gemini-3-pro-image-preview"

# Default file paths (relative to where script is run, usually project root)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
INPUT_FILE = os.path.join(DATA_DIR, "batch_input.json")
STAGING_FILE = os.path.join(DATA_DIR, "batch_staging.json")
JOB_ID_FILE = os.path.join(DATA_DIR, "latest_batch_job.txt")

def get_client():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment")
    return genai.Client(api_key=API_KEY)

def submit_batch_job(staging_path: str = STAGING_FILE) -> Optional[str]:
    """
    Submits an image batch job using inline requests.
    Returns the Job Name (ID) if successful, None otherwise.
    """
    client = get_client()
    
    # Load staging data
    print(f"Reading staging data from {staging_path}...")
    try:
        with open(staging_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
    except FileNotFoundError:
        print(f"Error: {staging_path} not found.")
        return None
    
    # Create inline requests with response_modalities
    inline_requests = []
    for i, item in enumerate(items):
        visual_prompt = item.get("visual_prompt")
        theme = item.get("theme", "Standard Mnemonic")
        visual_style = item.get("visual_style", "cartoon")
        
        # Construct the image generation instruction
        image_gen_instruction = prompts.get_image_generation_prompt(
            visual_prompt, theme, visual_style
        )
        
        # Create inline request with response_modalities
        inline_requests.append({
            'contents': [{'parts': [{'text': image_gen_instruction}]}],
            'config': {
                'response_modalities': ['TEXT', 'IMAGE'],  # CRITICAL for images!
                'image_config': {'aspect_ratio': '4:3'}
            }
        })
    
    print(f"Submitting {len(inline_requests)} image requests via inline batch...")
    
    # Submit batch job with inline requests
    try:
        batch_job = client.batches.create(
            model=MODEL_NAME,
            src=inline_requests,  # Inline requests, not file upload!
            config=types.CreateBatchJobConfig(
                display_name=f"medmonics_images_{int(time.time())}",
            )
        )
        print(f"✅ Job Created: {batch_job.name}")
        print(f"   State: {batch_job.state.name}")
        
        # Save Job ID
        with open(JOB_ID_FILE, "w") as f:
            f.write(batch_job.name)
        print(f"   Job ID saved to {JOB_ID_FILE}")
        
        return batch_job.name
        
    except Exception as e:
        print(f"❌ Batch submission failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    job_name = submit_batch_job()
    if job_name:
        print(f"\n🎉 Batch job submitted successfully!")
        print(f"   Job ID: {job_name}")
        print(f"   Use batch_retrieve.py to check status and retrieve results.")
    else:
        print(f"\n❌ Batch job submission failed.")
        sys.exit(1)
