"""
MedMonics Generation Pipeline Module

This module contains the core 5-step generation pipeline for creating medical mnemonics
with visual aids, character detection, and quiz questions.

Pipeline Steps:
    1. Generate Mnemonic (step1_generate_mnemonic)
       - Input: topic, language, theme, visual_style
       - Output: MnemonicResponse (topic, facts, story, associations, visualPrompt)
       - Uses character-based puns to create memorable associations
       
    2. Enhance Visual Prompt (step2_enhance_visual_prompt)
       - Input: MnemonicResponse, theme
       - Output: Enhanced visual prompt string
       - Refines description for better image generation
       
    3. Generate Image (step3_generate_image)
       - Input: enhanced_visual_prompt, theme, visual_style
       - Output: Image bytes (PNG)
       - Falls back to simpler theme if generation fails
       
    4. Analyze Bounding Boxes (step4_analyze_bboxes)
       - Input: image_bytes, MnemonicResponse
       - Output: BboxAnalysisResponse (character positions)
       - Detects 2D bounding boxes for character highlighting
       
    5. Generate Quiz (step5_generate_quiz)
       - Input: MnemonicResponse, language
       - Output: QuizList (questions linked to characters)
       - Creates multiple-choice questions for each association

Batch Processing Methods:
    - generate_breakdown_markdown(): Breaks down topics into subtopics
    - parse_markdown_to_items(): Parses markdown into batch input items

Data Models:
    - Association: Character-to-medical-term mapping
    - MnemonicResponse: Complete mnemonic data
    - QuizItem: Single quiz question
    - QuizList: Collection of quiz questions
    - CharBox: Character bounding box
    - BboxAnalysisResponse: Collection of bounding boxes

Usage:
    from medmonics import MedMnemonicPipeline
    
    pipeline = MedMnemonicPipeline(api_key="your_key")
    
    # Full pipeline
    mnemonic = pipeline.step1_generate_mnemonic("Cushing's Syndrome", "en", "Standard")
    visual = pipeline.step2_enhance_visual_prompt(mnemonic, "Standard")
    image = pipeline.step3_generate_image(visual, "Standard", "cartoon")
    bboxes = pipeline.step4_analyze_bboxes(image, mnemonic)
    quiz = pipeline.step5_generate_quiz(mnemonic, "en")
"""

import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from . import prompts

# Load variables from .env
load_dotenv()

# --- Pydantic Models for Schema Enforcement ---

class Association(BaseModel):
    character: str
    medicalTerm: str
    explanation: Optional[str] = None

class MnemonicResponse(BaseModel):
    topic: str
    facts: Optional[List[str]] = None
    story: str
    associations: List[Association]
    visualPrompt: str

class QuizItem(BaseModel):
    character: Optional[str] = None
    question: str
    options: List[str]
    correctOptionIndex: int
    explanation: str

class QuizList(BaseModel):
    quizzes: List[QuizItem]

class CharBox(BaseModel):
    character: str
    box_2d: List[int]

class BboxAnalysisResponse(BaseModel):
    boxes: List[CharBox]

# --- Pipeline Class ---

class MedMnemonicPipeline:
    """
    Main orchestrator for the MedMonics 5-step generation pipeline.
    
    This class manages the complete workflow from topic input to final mnemonic
    with illustration, character detection, and quiz questions.
    
    Attributes:
        api_key (str): Google Gemini API key
        client (genai.Client): Initialized Gemini API client
    
    Methods:
        step1_generate_mnemonic(): Create mnemonic story and associations
        step2_enhance_visual_prompt(): Refine visual description
        step3_generate_image(): Generate illustration
        step4_analyze_bboxes(): Detect character positions
        step5_generate_quiz(): Create quiz questions
        generate_breakdown_markdown(): Break down topics for batch processing
        parse_markdown_to_items(): Parse breakdown into batch items
    
    Example:
        >>> pipeline = MedMnemonicPipeline()
        >>> result = pipeline.step1_generate_mnemonic(
        ...     topic="Cushing's Syndrome symptoms",
        ...     language="en",
        ...     theme="Standard Mnemonic",
        ...     visual_style="cartoon"
        ... )
        >>> print(result.story)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or passed to constructor")
        self.client = genai.Client(api_key=self.api_key)

    def step1_generate_mnemonic(self, topic: str, language: str, theme: str, visual_style: str = "cartoon") -> MnemonicResponse:
        system_prompt = prompts.get_mnemonic_prompt(language, theme, visual_style)
        response = self.client.models.generate_content(
            model=prompts.MODEL_FLASH,
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text=topic),
                    types.Part.from_text(text=system_prompt)
                ])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MnemonicResponse,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        return MnemonicResponse.model_validate_json(response.text)

    def step2_enhance_visual_prompt(self, mnemonic_data: MnemonicResponse, theme: str = "Standard Mnemonic") -> str:
        enhancement_prompt = prompts.get_regenerate_visual_prompt_prompt(
            topic=mnemonic_data.topic,
            story=mnemonic_data.story,
            associations=[a.model_dump() for a in mnemonic_data.associations],
            theme=theme
        )
        response = self.client.models.generate_content(
            model=prompts.MODEL_VISUAL_PROMPT,
            contents=[types.Content(parts=[types.Part.from_text(text=enhancement_prompt)])],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        return response.text

    def step3_generate_image(self, enhanced_visual_prompt: str, theme: str, visual_style: str = "cartoon") -> Optional[bytes]:
        def try_generate(current_theme: str) -> Optional[bytes]:
            image_gen_instruction = prompts.get_image_generation_prompt(enhanced_visual_prompt, current_theme, visual_style)
            try:
                img_response = self.client.models.generate_content(
                    model=prompts.MODEL_IMAGE_GEN,
                    contents=image_gen_instruction,
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(aspect_ratio="4:3")
                    )
                )
                
                if img_response.parts:
                    for part in img_response.parts:
                        if part.inline_data:
                            return part.inline_data.data
                return None
            except Exception as e:
                print(f"Error generating image with theme '{current_theme}': {e}")
                return None

        # Attempt 1: Requested Theme
        print(f"Attempting image generation with theme: '{theme}'")
        image_bytes = try_generate(theme)
        
        # Attempt 2: Safe Fallback
        if not image_bytes:
            SAFE_THEME = "Minimalist abstract medical vector art, blue and white, clean lines"
            print(f"Image generation failed. Retrying with safe theme: '{SAFE_THEME}'")
            image_bytes = try_generate(SAFE_THEME)

        if not image_bytes:
            print("Warning: All image generation attempts failed.")
            return None
            
        return image_bytes

    def step4_analyze_bboxes(self, image_bytes: Optional[bytes], mnemonic_data: MnemonicResponse) -> BboxAnalysisResponse:
        if not mnemonic_data.associations or not image_bytes:
            return BboxAnalysisResponse(boxes=[])
            
        targets_desc = "\n\n".join([
            f"- Target Character: \"{a.character}\"\n  Medical Concept: \"{a.medicalTerm}\"\n  Visual Description/Context: {a.explanation}"
            for a in mnemonic_data.associations
        ])
        bbox_prompt = prompts.get_bbox_analysis_prompt(targets_desc)
        
        bbox_response = self.client.models.generate_content(
            model=prompts.MODEL_FLASH,
            contents=[
                types.Content(parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    types.Part.from_text(text=bbox_prompt)
                ])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=BboxAnalysisResponse,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        return BboxAnalysisResponse.model_validate_json(bbox_response.text)

    def step5_generate_quiz(self, mnemonic_data: MnemonicResponse, language: str) -> QuizList:
        assoc_str_q = "\n".join([f"Character: {a.character} -> Medical Concept: {a.medicalTerm}" for a in mnemonic_data.associations])
        quiz_context = f"Topic: {mnemonic_data.topic}\nFacts: {mnemonic_data.facts}\nAssociations: {assoc_str_q}"
        quiz_prompt = prompts.get_quiz_prompt(quiz_context, language)
        
        quiz_response = self.client.models.generate_content(
            model=prompts.MODEL_FLASH,
            contents=[
                types.Content(parts=[
                    types.Part.from_text(text=quiz_context),
                    types.Part.from_text(text=quiz_prompt)
                ])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QuizList,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            )
        )
        return QuizList.model_validate_json(quiz_response.text)

    def generate_breakdown_markdown(self, input_data: Any, input_type: str = "topic", language: str = "en") -> str:
        """
        Generates a markdown breakdown of the topic or content.
        input_type: "topic" (input_data is str) or "content" (input_data is bytes/file)
        """
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="low")
        )

        if input_type == "topic":
            topic = str(input_data)
            prompt = prompts.get_topic_breakdown_prompt(topic, language)
            contents = [types.Content(parts=[types.Part.from_text(text=prompt)])]
        else: # content
            prompt = prompts.get_content_breakdown_prompt(language)
            # input_data should be bytes. If it's a file path, read it. 
            # Check if it's bytes or Part.
            file_part = types.Part.from_bytes(data=input_data, mime_type="application/pdf") # Default to PDF or adjust
            # If input_data is a Part just use it? The caller should handle. 
            # Let's assume input_data is bytes and we default to PDF/Text for now or assume text part.
            # Actually, let's keep it simple: input_data IS the bytes, we assume PDF for now as general doc. 
            # Or we can accept a part.
            
            contents = [
                types.Content(parts=[
                    file_part, 
                    types.Part.from_text(text=prompt)
                ])
            ]

        response = self.client.models.generate_content(
            model=prompts.MODEL_FLASH, # Using Flash 3 Preview
            contents=contents,
            config=config
        )
        return response.text

    def parse_markdown_to_items(self, markdown_text: str, language: str, original_input: str = "Unknown Source", visual_style: str = "cartoon") -> List[Dict[str, Any]]:
        """
        Parses the markdown output to extract subtopics for batch input.
        Schema: { "input": original_input, "title": subtopic_name, "topic": content, "language": language, "visual_style": visual_style }
        """
        # Cleanup: Remove markdown code blocks if present
        markdown_text = markdown_text.replace("```markdown", "").replace("```", "").strip()
        
        items = []
        lines = markdown_text.split('\n')
        current_topic_name = "Introduction" # This will be the 'title' in the JSON
        current_content = []
        
        # We ignore the # H1 title for the individual item storage as per latest request,
        # but we use original_input as the 'input' box.
        
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                # Extract first H1 as a potential context or ignored? 
                # User wants "input: sepsis; title: introduction; topic: definition"
                # So we skip H1 and use ## as 'title'
                continue
            elif line.startswith("## "):
                # Save previous
                if current_content:
                    items.append({
                        "input": original_input,
                        "title": current_topic_name,
                        "topic": f"{' '.join(current_content)[:1000]}",
                        "language": language,
                        "visual_style": visual_style
                    })
                
                # Start new subtopic
                current_topic_name = line[3:].strip().strip("[]")
                current_content = []
            else:
                if line:
                    current_content.append(line)
        
        # Add last one
        if current_topic_name and current_content:
             items.append({
                "input": original_input,
                "title": current_topic_name,
                "topic": f"{' '.join(current_content)[:1000]}",
                "language": language,
                "visual_style": visual_style
            })
            
        return items
