import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import base64
from typing import Optional, Dict, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from medmonics.pipeline import MedMnemonicPipeline, MnemonicResponse, BboxAnalysisResponse, QuizItem, QuizList

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# File paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STAGING_FILE = os.path.join(DATA_DIR, "batch_staging.json")
JOB_ID_FILE = os.path.join(DATA_DIR, "latest_batch_job.txt")
STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'generations')

def get_client():
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment")
    return genai.Client(api_key=API_KEY)

def check_batch_status(job_name: str = None) -> Dict:
    """
    Check the status of a batch job.
    Returns a dict with 'state', 'message', and other metadata.
    """
    client = get_client()
    
    # Resolve job name
    if not job_name:
        try:
            with open(JOB_ID_FILE, 'r') as f:
                job_name = f.read().strip()
        except FileNotFoundError:
            return {
                "state": "ERROR",
                "message": f"Job ID file not found: {JOB_ID_FILE}"
            }
    
    print(f"Checking status for job: {job_name}")
    
    try:
        batch_job = client.batches.get(name=job_name)
        
        status_info = {
            "state": batch_job.state.name,
            "job_name": batch_job.name,
            "display_name": batch_job.display_name,
        }
        
        # Add request/response counts if available
        if hasattr(batch_job, 'request_count'):
            status_info["request_count"] = batch_job.request_count
        
        if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
            status_info["message"] = "Job completed successfully!"
            
        elif batch_job.state.name == 'JOB_STATE_FAILED':
            error_msg = getattr(batch_job, 'error', 'Unknown Error')
            status_info["message"] = f"Job Failed. Error: {error_msg}"
            
        elif batch_job.state.name == 'JOB_STATE_CANCELLED':
            status_info["message"] = "Job was cancelled."
            
        elif batch_job.state.name == 'JOB_STATE_RUNNING':
            status_info["message"] = "Job is still running..."
            
        else:
            status_info["message"] = f"Job state: {batch_job.state.name}"
        
        return status_info
        
    except Exception as e:
        return {
            "state": "ERROR",
            "message": f"Failed to get job status: {e}"
        }

def retrieve_and_finalize(job_name: str = None, storage_backend=None) -> int:
    """
    Retrieve batch results from inlined_responses and finalize them.
    Returns the number of items successfully processed.
    If storage_backend is provided, saves using that backend. Otherwise saves locally.
    """
    client = get_client()
    
    # Resolve job name
    if not job_name:
        try:
            with open(JOB_ID_FILE, 'r') as f:
                job_name = f.read().strip()
        except FileNotFoundError:
            print(f"Error: Job ID file not found: {JOB_ID_FILE}")
            return 0
    
    print(f"Retrieving results for job: {job_name}")
    
    # Get job
    try:
        batch_job = client.batches.get(name=job_name)
    except Exception as e:
        print(f"Error getting job: {e}")
        return 0
    
    # Check if job succeeded
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        print(f"Job not ready: {batch_job.state.name}")
        print(f"Please wait for job to complete before retrieving results.")
        return 0
    
    # Load staging data
    if not os.path.exists(STAGING_FILE):
        print(f"Error: Staging file not found: {STAGING_FILE}")
        return 0
    
    with open(STAGING_FILE, 'r', encoding='utf-8') as f:
        staging_items = json.load(f)
    
    print(f"Found {len(staging_items)} staged items")
    
    # Check if we have inlined_responses
    if not hasattr(batch_job.dest, 'inlined_responses'):
        print("Error: Batch job has no inlined_responses")
        print("This might be because the job was submitted with file upload instead of inline requests")
        return 0
    
    inlined_responses = batch_job.dest.inlined_responses
    print(f"Found {len(inlined_responses)} responses")
    
    # Process each response
    pipeline = MedMnemonicPipeline(api_key=API_KEY)
    count = 0
    
    for i, inline_response in enumerate(inlined_responses):
        if i >= len(staging_items):
            print(f"Warning: More responses than staged items, skipping response {i}")
            break
        
        staged = staging_items[i]
        topic = staged.get('mnemonic_data', {}).get('topic', f'item-{i}')
        
        # Check for errors
        if inline_response.error:
            print(f"❌ Error for item {i} ({topic[:50]}): {inline_response.error}")
            continue
        
        # Extract image from response
        image_bytes = None
        if inline_response.response and inline_response.response.candidates:
            for part in inline_response.response.candidates[0].content.parts:
                if part.inline_data:
                    # inline_data.data already contains raw bytes, no decoding needed
                    image_bytes = part.inline_data.data
                    print(f"✅ Found image for item {i} ({topic[:50]}): {len(image_bytes) // 1024} KB")
                    break
        
        if not image_bytes:
            print(f"❌ No image found for item {i} ({topic[:50]})")
            continue
        
        # Reconstruct models from staging data
        try:
            mnemonic_data = MnemonicResponse(**staged["mnemonic_data"])
            # Quiz data is stored as {"quizzes": [...]}
            quiz_items = [QuizItem(**q) for q in staged["quiz_data"]["quizzes"]]
        except Exception as e:
            print(f"❌ Error reconstructing data for item {i}: {e}")
            continue
        
        # Run bbox analysis (Step 4)
        print(f"🔍 Running Bbox analysis for: {mnemonic_data.topic[:50]}...")
        try:
            bbox_data = pipeline.step4_analyze_bboxes(image_bytes, mnemonic_data)
        except Exception as e:
            print(f"⚠️  Bbox analysis failed for item {i}, using empty bbox data: {e}")
            bbox_data = BboxAnalysisResponse(boxes=[])
        
        specialty = staged.get("input", "Batch_Import")
        
        if storage_backend:
            # Use provided storage backend (GCS, etc)
            try:
                # Need QuizList object
                quiz_list = QuizList(quizzes=quiz_items)
                path = storage_backend.save_generation(
                    mnemonic_data=mnemonic_data,
                    bbox_data=bbox_data,
                    quiz_data=quiz_list,
                    image_bytes=image_bytes,
                    specialty=specialty
                )
                print(f"☁️ Saved to Cloud/Storage: {path}")
                count += 1
            except Exception as e:
                print(f"❌ Error saving to storage backend: {e}")
        else:
            # Save to standard local location (Legacy/Script usage)
            timestamp = int(time.time() * 1000)
            specialty_slug = specialty.replace(" ", "_").lower()
            folder_name = f"{timestamp}_{i}_{mnemonic_data.topic[:30].replace(' ', '_')}"
            
            final_folder = os.path.join(STORAGE_DIR, specialty_slug, folder_name)
            os.makedirs(final_folder, exist_ok=True)
            
            # Save data.json
            all_data = {
                "mnemonic_data": mnemonic_data.model_dump(),
                "bbox_data": bbox_data.model_dump(),
                "quiz_data": {"quizzes": [q.model_dump() for q in quiz_items]},
                "metadata": {
                    "topic_id": f"batch-{i}",
                    "timestamp": timestamp,
                    "specialty": specialty,
                    "batch_job": job_name
                }
            }
            
            with open(os.path.join(final_folder, "data.json"), "w", encoding="utf-8") as df:
                json.dump(all_data, df, indent=2, ensure_ascii=False)
            
            # Save image
            with open(os.path.join(final_folder, "image.png"), "wb") as imf:
                imf.write(image_bytes)
            
            print(f"💾 Saved to: {final_folder}")
            count += 1
    
    print(f"\n🎉 Finalized {count}/{len(staging_items)} items successfully!")
    return count

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrieve and process batch job results')
    parser.add_argument('--job-name', type=str, help='Batch job name (optional, uses latest if not provided)')
    parser.add_argument('--status-only', action='store_true', help='Only check status, do not retrieve')
    
    args = parser.parse_args()
    
    if args.status_only:
        # Just check status
        status = check_batch_status(args.job_name)
        print(f"\n{'='*70}")
        print(f"Job Status: {status['state']}")
        print(f"Message: {status['message']}")
        if 'display_name' in status:
            print(f"Display Name: {status['display_name']}")
        if 'request_count' in status:
            print(f"Request Count: {status['request_count']}")
        print(f"{'='*70}")
    else:
        # Retrieve and finalize
        count = retrieve_and_finalize(args.job_name)
        if count > 0:
            print(f"\n✅ Successfully processed {count} items!")
        else:
            print(f"\n❌ No items were processed.")
            sys.exit(1)
