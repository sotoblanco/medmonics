"""
MedMonics Batch Data Loader Module

This module provides utilities for loading and normalizing batch API results from
Gemini's Batch API. It handles inconsistent JSON key formats from different model
responses and links batch results to their original input topics.

Key Functions:
    normalize_keys(): Normalizes inconsistent JSON keys to standard format
        - Handles Spanish/English key variations
        - Maps nested structures to flat format
        - Provides fallback values for missing fields
        - Standard output keys: topic, story, associations, visual_prompt, quiz, facts
        
    parse_jsonl_results(): Parses Gemini Batch API JSONL output files
        - Reads line-by-line JSONL format
        - Extracts text from nested response structure
        - Cleans markdown code fences
        - Links results to input topics via custom_id
        - Filters out error responses
        
Expected Input Format (JSONL):
    Each line is a JSON object with structure:
    {
        "custom_id": "req-0-1234567890",
        "response": {
            "candidates": [{
                "content": {
                    "parts": [{"text": "...JSON content..."}]
                }
            }]
        }
    }

Expected Output Format (Normalized):
    {
        "custom_id": "req-0-1234567890",
        "topic": "Medical topic name",
        "story": "Mnemonic story text",
        "associations": [
            {
                "character": "Character name",
                "medical_term": "Medical concept",
                "explanation": "Why this character represents this concept"
            }
        ],
        "visual_prompt": "Visual description for image generation",
        "quiz": [
            {
                "question": "Quiz question text",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Correct answer text or index"
            }
        ],
        "facts": ["Fact 1", "Fact 2", ...],
        "input_title": "Original input subtopic title",
        "input_topic": "Original input topic content"
    }

Usage:
    from medmonics.data_loader import parse_jsonl_results
    
    results = parse_jsonl_results(
        file_path="data/batch_output.jsonl",
        input_file_path="data/batch_input.json"
    )
    
    for item in results:
        print(f"Topic: {item['topic']}")
        print(f"Story: {item['story']}")
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

def normalize_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes keys from inconsistent JSON responses to a standard format.
    """
    normalized = {}
    
    # Story
    story_keys = ["mnemonico_historia", "mnemonic_story", "mnemotecnia_historia", "titulo_historia", "historia_mnemonica", "mnemotecnico_historia"]
    for k in story_keys:
        if k in data:
            if isinstance(data[k], dict) and "historia" in data[k]:
                 normalized["story"] = data[k]["historia"] # Handle nested case
            else:
                 normalized["story"] = data[k]
            break
            
    # Handle the nested story case more robustly if not found
    if "story" not in normalized:
        if "mnemonico" in data and isinstance(data["mnemonico"], dict) and "historia" in data["mnemonico"]:
            normalized["story"] = data["mnemonico"]["historia"]
        elif "mnemotecnia" in data and isinstance(data["mnemotecnia"], dict) and "historia" in data["mnemotecnia"]:
            normalized["story"] = data["mnemotecnia"]["historia"]
    
    # Associations
    assoc_keys = ["asociaciones", "associations"]
    for k in assoc_keys:
        if k in data:
            normalized["associations"] = []
            for item in data[k]:
                norm_item = {}
                # Character
                char_keys = ["personaje", "character", "personaje_elemento", "personaje_objeto"]
                for ck in char_keys:
                    if ck in item:
                        norm_item["character"] = item[ck]
                        break
                # Medical Term
                term_keys = ["termino_medico", "medical_term", "elemento_medico"]
                for tk in term_keys:
                    if tk in item:
                        norm_item["medical_term"] = item[tk]
                        break
                
                # Explanation
                expl_keys = ["explicacion", "explanation", "descripcion"]
                for ek in expl_keys:
                    if ek in item:
                        norm_item["explanation"] = item[ek]
                        break
                
                normalized["associations"].append(norm_item)
            break
            
    # Visual Prompt
    prompt_keys = ["prompt_visual", "visual_prompt"]
    for k in prompt_keys:
        if k in data:
            normalized["visual_prompt"] = data[k]
            break
            
    # Quiz
    quiz_keys = ["quiz", "cuestionario", "quiz_preguntas", "preguntas_quiz", "cuestionario_final"]
    for k in quiz_keys:
        if k in data:
            normalized["quiz"] = []
            for q in data[k]:
                norm_q = {}
                # Question
                quest_keys = ["pregunta", "question"]
                for qk in quest_keys:
                    if qk in q:
                        norm_q["question"] = q[qk]
                        break
                # Options
                opt_keys = ["opciones", "options"]
                for ok in opt_keys:
                    if ok in q:
                        norm_q["options"] = q[ok]
                        break
                # Answer
                ans_keys = ["respuesta_correcta", "answer", "respuesta", "correct_answer"]
                for ak in ans_keys:
                    if ak in q:
                        norm_q["answer"] = q[ak]
                        break
                
                normalized["quiz"].append(norm_q)
            break

    # Title/Topic
    title_keys = ["topic", "titulo", "tema"]
    for k in title_keys:
         if k in data:
            normalized["topic"] = data[k]
            break

    # Facts
    fact_keys = ["facts", "datos", "hechos", "puntos_clave"]
    for k in fact_keys:
        if k in data:
            normalized["facts"] = data[k]
            break
            
    return normalized


def parse_jsonl_results(file_path: str, input_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    results = []
    path = Path(file_path)
    if not path.exists():
        return results

    # Load inputs if provided
    input_data = []
    if input_file_path:
        input_path = Path(input_file_path)
        if input_path.exists():
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    input_data = json.load(f)
            except Exception as e:
                print(f"Error loading input file: {e}")

    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            
            try:
                # Outer JSON
                outer_obj = json.loads(line)
                
                custom_id = outer_obj.get("custom_id", f"unknown-{line_num}")
                
                # Check for errors first
                if "error" in outer_obj:
                    # Skip error lines
                    continue

                if "response" in outer_obj and "candidates" in outer_obj["response"]:
                     candidates = outer_obj["response"]["candidates"]
                     if candidates and len(candidates) > 0:
                         content = candidates[0].get("content", {})
                         parts = content.get("parts", [])
                         if parts and len(parts) > 0:
                             raw_text = parts[0].get("text", "")
                             
                             # Clean code fences
                             raw_text = raw_text.strip()
                             if raw_text.startswith("```"):
                                 lines = raw_text.splitlines()
                                 # Simply remove the first and last line which are usually ```json and ```
                                 if len(lines) >= 3:
                                     # Check if first line is ```json or similar
                                     if lines[0].startswith("```"):
                                         raw_text = "\n".join(lines[1:-1])
                                 else:
                                     # Edge case: small content
                                     raw_text = raw_text.replace("```json", "").replace("```", "")
                             
                             try:
                                 inner_data = json.loads(raw_text)
                                 normalized_item = normalize_keys(inner_data)
                                 normalized_item["custom_id"] = custom_id
                                 
                                 # --- LINK INPUT TOPIC ---
                                 if custom_id and custom_id.startswith("req-"):
                                     try:
                                         # format: req-{index}-{timestamp}
                                         parts_id = custom_id.split("-")
                                         if len(parts_id) >= 2:
                                             idx = int(parts_id[1])
                                             if 0 <= idx < len(input_data):
                                                 # Prefer title for display, fallback to topic
                                                 normalized_item["input_title"] = input_data[idx].get("title", "")
                                                 normalized_item["input_topic"] = input_data[idx].get("topic", "")
                                     except:
                                         pass

                                 # If topic is missing, try to infer or use ID
                                 if "topic" not in normalized_item:
                                    # Try to find it in other fields
                                    if "associations" in normalized_item and normalized_item["associations"]:
                                        # Use first medical term
                                        first_assoc = normalized_item["associations"][0]
                                        if "medical_term" in first_assoc:
                                            normalized_item["topic"] = first_assoc["medical_term"] + " (Inferred)"
                                    
                                    # Try story text fallback (first few words)
                                    if "topic" not in normalized_item and "story" in normalized_item:
                                         story_start = normalized_item["story"][:30].strip() + "..."
                                         normalized_item["topic"] = f"Story: {story_start}"
                                         
                                    if "topic" not in normalized_item:
                                        normalized_item["topic"] = f"Result {custom_id}"

                                 results.append(normalized_item)
                             except json.JSONDecodeError:
                                 # inner parsing failed
                                 pass
            except json.JSONDecodeError:
                # outer parsing failed
                pass
                
    return results
