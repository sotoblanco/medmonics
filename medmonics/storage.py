from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import re
import io
import uuid

# GCS imports
try:
    from google.cloud import storage
    from google.oauth2 import service_account
    import streamlit as st
except ImportError:
    storage = None

def slugify(text):
    """Helper to create safe directory/file names."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text[:30]

class StorageBackend(ABC):
    """Abstract base class for MedMnemonic storage."""

    @abstractmethod
    def save_generation(self, mnemonic_data: Any, bbox_data: Any, quiz_data: Any, 
                       image_bytes: bytes, specialty: str = "General", parent_id: str = None) -> str:
        """
        Saves the generation data and checks for duplicates/updates.
        Returns the identifier (path or key) of the saved item.
        """
        pass

    @abstractmethod
    def list_generations(self, specialty_filter: str = None) -> List[dict]:
        """
        Returns a list of available generations. 
        Each item dict should minimally contain:
        - 'name' or 'id'
        - 'timestamp'
        - 'specialty' (inferred or explicit)
        - 'identifier' (to pass to load_generation)
        """
        pass

    @abstractmethod
    def load_generation(self, identifier: str) -> Tuple[Any, Any, Any, bytes, dict]:
        """
        Loads a generation by its identifier.
        Returns (mnemonic_data, bbox_data, quiz_data, image_bytes, metadata)
        """
        pass

class LocalStorage(StorageBackend):
    """Legacy local filesystem storage."""
    
    def __init__(self, base_dir: str = "generations"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

    def save_generation(self, mnemonic_data, bbox_data, quiz_data, image_bytes, specialty="General", parent_id=None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = slugify(mnemonic_data.topic)
        
        # Create specialty subfolder
        specialty_slug = slugify(specialty) if specialty else "general"
        specialty_folder = self.base_dir / specialty_slug
        specialty_folder.mkdir(parents=True, exist_ok=True)
        
        folder_path = specialty_folder / f"{timestamp}_{topic_slug}"
        folder_path.mkdir(parents=True, exist_ok=True)
        
        # Save Image
        with open(folder_path / "image.png", "wb") as f:
            f.write(image_bytes)
        
        # Generate unique ID for this generation
        topic_id = str(uuid.uuid4())
        
        # Save Data
        all_data = {
            "mnemonic_data": mnemonic_data.model_dump(),
            "bbox_data": bbox_data.model_dump(),
            "quiz_data": quiz_data.model_dump(),
            "metadata": {
                "timestamp": timestamp,
                "topic": mnemonic_data.topic,
                "specialty": specialty,
                "topic_id": topic_id,
                "parent_id": parent_id
            }
        }
        with open(folder_path / "data.json", "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        return str(folder_path)

    def list_generations(self, specialty_filter=None):
        if not self.base_dir.exists():
            return []
        
        all_folders = []
        
        # Scan all specialty subfolders
        for item in self.base_dir.iterdir():
            if item.is_dir():
                # Check if it's a specialty folder (contains subfolders with data.json)
                sub_items = list(item.iterdir())
                # Only check sub_items that are directories to avoid errors
                dirs_in_sub = [s for s in sub_items if s.is_dir()]
                
                if dirs_in_sub and any((s / "data.json").exists() for s in dirs_in_sub):
                    # It's a specialty folder
                    if specialty_filter is None or item.name == slugify(specialty_filter):
                        for gen_folder in item.iterdir():
                            if gen_folder.is_dir() and (gen_folder / "data.json").exists():
                                all_folders.append(gen_folder)
                elif (item / "data.json").exists():
                    # Legacy: direct generation folder (no specialty)
                    if specialty_filter is None or specialty_filter == "General":
                        all_folders.append(item)
        
        # Sort by timestamp (prefix/name) descending
        # Return dictionaries to match interface
        results = []
        sorted_folders = sorted(all_folders, key=lambda x: x.name, reverse=True)
        
        for folder in sorted_folders:
            results.append({
                "name": folder.name,
                "identifier": str(folder),
                "timestamp": folder.name.split('_')[0] if '_' in folder.name else "",
                # We could peek metadata here but it might be slow for many files
            })
            
        return results

    def load_generation(self, identifier):
        folder_path = Path(identifier)
        if not folder_path.exists():
            raise FileNotFoundError(f"Generation not found at {identifier}")

        with open(folder_path / "data.json", "r", encoding="utf-8") as f:
            all_data = json.load(f)
        
        with open(folder_path / "image.png", "rb") as f:
            image_bytes = f.read()
        
        # We process Pydantic conversions in the app layer or here?
        # The interface says "Any", usually returning dicts is safer for Storage
        # decoupling, but for now let's return the raw loaded dicts + bytes
        # to allow app.py to reconstruct models as it does now.
        
        return (
            all_data["mnemonic_data"], 
            all_data["bbox_data"], 
            all_data["quiz_data"], 
            image_bytes, 
            all_data.get("metadata", {})
        )


class GCSBackend(StorageBackend):
    """Google Cloud Storage backend."""
    
    def __init__(self, bucket_name: str, service_account_info: dict = None):
        if storage is None:
            raise ImportError("google-cloud-storage is not installed. Add it to requirements.txt")
        
        if service_account_info:
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            self.client = storage.Client(credentials=credentials)
        else:
            # Fallback to default/env var credentials
            self.client = storage.Client()
            
        self.bucket_name = bucket_name
        self.bucket = self.client.bucket(bucket_name)

    def save_generation(self, mnemonic_data, bbox_data, quiz_data, image_bytes, specialty="General", parent_id=None) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        topic_slug = slugify(mnemonic_data.topic)
        specialty_slug = slugify(specialty) if specialty else "general"
        
        # Structure: specialty/timestamp_topic/
        base_path = f"{specialty_slug}/{timestamp}_{topic_slug}"
        
        # 1. Save Image
        image_blob = self.bucket.blob(f"{base_path}/image.png")
        image_blob.upload_from_string(image_bytes, content_type="image/png")
        
        # 2. Prepare Data
        topic_id = str(uuid.uuid4())
        all_data = {
            "mnemonic_data": mnemonic_data.model_dump(),
            "bbox_data": bbox_data.model_dump(),
            "quiz_data": quiz_data.model_dump(),
            "metadata": {
                "timestamp": timestamp,
                "topic": mnemonic_data.topic,
                "specialty": specialty,
                "topic_id": topic_id,
                "parent_id": parent_id
            }
        }
        
        # 3. Save JSON
        json_blob = self.bucket.blob(f"{base_path}/data.json")
        json_blob.upload_from_string(
            json.dumps(all_data, indent=2, ensure_ascii=False),
            content_type="application/json"
        )
        
        return base_path

    def list_generations(self, specialty_filter=None):
        # We need to list prefixes effectively.
        # Structure is {specialty_slug}/{folder}/data.json
        
        prefix = None
        if specialty_filter:
            prefix = f"{slugify(specialty_filter)}/"
        
        # List blobs. GCS doesn't have true directories, but we can emulate.
        # A more efficient way is to list blobs with delimiter if we had a flat structure,
        # but here we might want to list all 'data.json' files.
        # Or, strictly list the specialty folders if no filter, or contents if filter.
        
        # Strategy: List all 'data.json' files under the prefix
        # This might be slow if there are thousands, but acceptable for now.
        
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix, match_glob="**/data.json")
        
        results = []
        for blob in blobs:
            # blob.name might be "cardiology/20260207_123000_heart/data.json"
            parts = blob.name.split("/")
            if len(parts) >= 2:
                # folder_name is usually the 2nd to last part
                folder_name = parts[-2]
                path_prefix = "/".join(parts[:-1]) # remove data.json
                
                results.append({
                    "name": folder_name,
                    "identifier": path_prefix,
                    "timestamp": folder_name.split('_')[0] if '_' in folder_name else "",
                })
        
        # Sort by name (timestamp) descending
        return sorted(results, key=lambda x: x["name"], reverse=True)

    def load_generation(self, identifier):
        # identifier is the GCS prefix folder, e.g. "cardiology/2026_..._topic"
        
        json_blob = self.bucket.blob(f"{identifier}/data.json")
        image_blob = self.bucket.blob(f"{identifier}/image.png")
        
        if not json_blob.exists():
             raise FileNotFoundError(f"GCS path not found: {identifier}/data.json")
             
        json_str = json_blob.download_as_text(encoding="utf-8")
        all_data = json.loads(json_str)
        
        image_bytes = image_blob.download_as_bytes()
        
        return (
            all_data["mnemonic_data"], 
            all_data["bbox_data"], 
            all_data["quiz_data"], 
            image_bytes, 
            all_data.get("metadata", {})
        )
