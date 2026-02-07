"""
MedMonics Prompt Templates Module

This module contains all prompt templates and model configurations for the MedMonics
AI-powered mnemonic generator.

Model Constants:
    MODEL_FLASH: Fast text generation model (gemini-3-flash-preview)
    MODEL_VISUAL_PROMPT: Visual prompt enhancement model
    MODEL_IMAGE_GEN: Image generation model (gemini-3-pro-image-preview)
    MODEL_TTS: Text-to-speech model

Language Support:
    - English (en): Default language
    - Spanish (es): Full support with culturally appropriate character names

Key Functions:
    - get_mnemonic_prompt(): Main mnemonic generation prompt (Step 1)
    - get_image_generation_prompt(): Image generation instructions (Step 3)  
    - get_bbox_analysis_prompt(): Character detection prompt (Step 4)
    - get_quiz_prompt(): Quiz generation prompt (Step 5)
    - get_topic_breakdown_prompt(): Batch topic breakdown
    - get_visual_style_instruction(): Style-specific visual instructions

Visual Styles:
    - cartoon: Vibrant 3D Chibi/Pixar style (default)
    - photorealistic: Cinematic National Geographic photography
    - professional: Corporate headshot aesthetic

Usage:
    from medmonics import prompts
    
    prompt = prompts.get_mnemonic_prompt(
        language="en",
        theme="Standard Mnemonic",
        visual_style="cartoon"
    )
"""

from typing import List, Dict, Any
import json

# --- Model Constants ---
MODEL_FLASH = "gemini-3-flash-preview"
MODEL_VISUAL_PROMPT = "gemini-3-flash-preview"
MODEL_IMAGE_GEN = "gemini-3-pro-image-preview"
MODEL_TTS = "gemini-2.5-flash-preview-tts"

# --- Language Instructions ---
LANGUAGE_INSTRUCTION_ES = """
        IMPORTANT: OUTPUT MUST BE IN SPANISH (ESPAÑOL).
        - ALL values, text, descriptions, story content, explanations, and terms MUST be in Spanish.
        - The characters should have Spanish names or names that make sense in a Spanish pun context.
        """
LANGUAGE_INSTRUCTION_EN = "Provide all output in English."

def get_language_instruction(lang: str) -> str:
    if lang == 'es':
        return LANGUAGE_INSTRUCTION_ES
    return LANGUAGE_INSTRUCTION_EN

# --- Prompts ---

def get_visual_style_instruction(style: str) -> str:
    """Returns specific visual instructions based on selected style."""
    styles = {
        "cartoon": "Style: Hyper-vibrant 3D Chibi/Pixar style with exaggerated expressions and cinematic colors.",
        "photorealistic": "Style: Cinematic, high-fidelity National Geographic photography. Real human beings with genuine skin textures, pores, and hair. Shot on 35mm lens, Kodak Portra 400 aesthetic. NO 3D renders, NO animation, NO cartoon elements.",
        "professional": "Style: Professional studio headshot, corporate/formal look. 85mm f/1.4 lens compression, classic three-point lighting, clean solid backdrop. Real people in business attire."
    }
    # Default to cartoon if style not recognized
    return styles.get(style.lower() if style else "cartoon", styles["cartoon"])

def get_mnemonic_prompt(language: str, theme: str = "Standard Mnemonic", visual_style: str = "cartoon") -> str:
    theme_instruction = f"The visual style and character setting should follow this theme: '{theme}'." if theme else ""
    visual_instr = get_visual_style_instruction(visual_style)
    
    tone_instr = "The tone should be humorous and wacky (cartoon style)."
    if visual_style.lower() == "photorealistic":
        tone_instr = "The tone should be cinematic and editorial (real people in dramatic or grounded scenes)."
    elif visual_style.lower() == "professional":
        tone_instr = "The tone should be professional and formal (corporate headshot aesthetic)."

    return f"""
    Act as an expert medical educator (like Picmonic or SketchyMedical).
    {get_language_instruction(language)}
    
    {theme_instruction}
    {visual_instr}
    
    1. Analyze the input to extract high-yield medical facts, dosages, symptoms, and treatments.
    2. Create a memorable mnemonic story to explain these facts. 
       - Use sound-alike characters (e.g., 'Macrolide' -> 'Macaroni Slide').
       - Keep language simple and narrative.
       - {tone_instr}
       - Describe characters as real human beings if the style is photorealistic.
    3. List the associations between characters and medical terms.
    4. Create a visual prompt for a high-quality illustration of this story.
       - IMPORTANT: The visual prompt must incorporate the theme: '{theme}'.

    Output a single JSON object.
    """

def get_regenerate_story_prompt(topic: str, facts: List[str], language: str, theme: str = "Standard Mnemonic") -> str:
    facts_str = "\n".join([f"- {f}" for f in facts])
    theme_instruction = f"Style/Theme: '{theme}'." if theme else ""
    return f"""
    {get_language_instruction(language)}
    {theme_instruction}
    Topic: {topic}
    Facts:
    {facts_str}

    Based on these SPECIFIC facts, generate:
    1. A wacky mnemonic story.
    2. The list of associations.
    3. A visual prompt for the image.
    
    Maintain the humorous, mnemonic style.
    """

def get_regenerate_visual_prompt_prompt(topic: str, story: str, associations: List[Any], theme: str = "Standard Mnemonic") -> str:
    # associations should be a list of dicts or objects with .dict()
    # We'll assume the caller passes the list of dicts or handles serialization before calling if complex
    
    # If associations are pydantic models, caller should dump them. 
    assoc_str = json.dumps(associations) if isinstance(associations, list) else str(associations)

    return f"""
    Topic: {topic}
    Story: {story}
    Associations: {assoc_str}
    Theme: {theme}

    Create a highly detailed visual description (visual prompt) for an image generator to illustrate this story. 
    Focus on visual clarity of the characters and consistency with the story's visual style.
    IMPORTANT: The visual prompt MUST follow and explicitly reference the theme: '{theme}'.
    """

def get_image_generation_prompt(visual_prompt: str, theme: str = None, visual_style: str = "cartoon") -> str:
    theme_suffix = f" Follow the style/aesthetic of '{theme}'." if theme else ""
    visual_instr = get_visual_style_instruction(visual_style)
    return f"""
    {visual_instr}
    
    Subject: {visual_prompt}. 
    {theme_suffix}
    Composition: A single cohesive scene. High quality, detailed."""

def get_bbox_analysis_prompt(targets_desc: str) -> str:
    return f"""
        You are an expert visual analyzer for medical mnemonic illustrations.
        
        Task: Identify the 2D bounding box for the specific characters listed below in the provided image.
        
        List of Targets:
        {targets_desc}

        Instructions:
        1. Analyze the image to locate the character described. Use the "Visual Description/Context" to disambiguate if necessary.
        2. Return the bounding box [ymin, xmin, ymax, xmax] for each character found.
        3. IMPORTANT: Use the normalized 0-1000 scale for coordinates (e.g., [450, 200, 600, 400]).
        4. If a character is not found, omit it from the list or return 0,0,0,0.
        
        Output Format:
        Return a JSON array of objects. Each object must have:
        - 'character': The exact "Target Character" name.
        - 'box_2d': [ymin, xmin, ymax, xmax].
        """

def get_quiz_prompt(context: str, language: str) -> str:
    return f"""
    {get_language_instruction(language)}
    Generate a challenging multiple-choice quiz based on the provided associations for a medical student audience.
    
    For each association listed above:
    1. Create a question that tests understanding of the medical concept.
       - Do NOT just ask "What does this character represent?".
       - Instead, ask about the *implication* of the fact (e.g., "What is the mechanism of action associated with this symbol?" or "What clinical presentation does this character signify?", "What is the treatment indicated by this symbol?").
       - If the association is simple, ask a second-order question related to that fact.
    Output Format:
    Return a JSON object with a 'quizzes' key containing an array of objects. Each object must have:
    - 'character': The exact "Target Character" name this question is about.
    - 'question': The question text.
    - 'options': An array of 4 options.
    - 'correctOptionIndex': The index of the correct answer (0-3).
    - 'explanation': A brief explanation.

    Generate questions for ALL associations.
    """


def get_topic_breakdown_prompt(topic: str, language: str) -> str:
    lang_instr = get_language_instruction(language)
    return f"""
    {lang_instr}
    Act as an expert medical educator specializing in creating comprehensive study materials for medical students.
    
    Topic: {topic}

    Task: Perform COMPREHENSIVE RESEARCH and breakdown this medical topic into detailed subtopics suitable for mnemonic creation.

    Critical Requirements:
    1.  **COMPREHENSIVE COVERAGE**: Research and include ALL major aspects of this topic.
       - Pathophysiology and mechanisms
       - Clinical presentations and signs/symptoms
       - Diagnostic criteria and tests
       - Treatment options and management
       - Complications and prognosis
       - Special populations or considerations
    
    2.  **GRANULAR SUBTOPICS**: Break down into MANY focused, specific subtopics.
       - Each subtopic should cover ONE specific concept
       - Create as many subtopics as necessary (do not limit yourself)
       - Aim for 6-10+ high-yield facts per subtopic
    
    3.  **HIGH-YIELD FACTS**: Focus on board-relevant, clinically important information.
       - Include specific values, criteria, timelines
       - Preserve exact classifications and staging systems
       - Include memorable associations and mnemonics triggers
       - Use precise medical terminology
    
    4.  **MNEMONIC-READY FORMAT**: Structure for optimal mnemonic creation.
       - List facts clearly and separately
       - Group related facts together
       - Use clear, memorable phrasing
    
    5.  **INCREMENTAL LEARNING**: Order logically from basics to advanced.
       - Start with foundational concepts (anatomy, physiology)
       - Progress through pathophysiology
       - Cover clinical presentation and diagnosis
       - End with treatment, complications, and prognosis

    Output Format (Strict Markdown):
    # [Comprehensive Descriptive Title for the Topic]
    
    ## [Subtopic 1: Specific Aspect Name]
    **Overview**: [Brief context for this aspect]
    
    **Key Facts** (for mnemonic creation):
    - [Fact 1 with specific details]
    - [Fact 2 with specific details]
    - [Fact 3 with specific details]
    - [... 6-10+ key high-yield facts]
    
    [Additional context or clinical pearls if helpful]

    ## [Subtopic 2: Next Specific Aspect]
    **Overview**: [Brief context]
    
    **Key Facts**:
    - [Comprehensive list of high-yield facts]
    
    [Continue for ALL important aspects...]
    
    REMEMBER: The goal is to create as many detailed subtopics as needed to ensure complete topic coverage. Each subtopic will become a separate mnemonic, so include ALL clinically important information!
    """

def get_content_breakdown_prompt(language: str) -> str:
    lang_instr = get_language_instruction(language)
    return f"""
    {lang_instr}
    Act as an expert medical educator specializing in creating comprehensive study materials for medical students.

    Task: Perform a THOROUGH and DETAILED analysis of the provided content to extract ALL important medical information for mnemonic creation.

    Critical Requirements:
    1.  **COMPREHENSIVE EXTRACTION**: Extract ALL key facts, mechanisms, clinical features, treatments, and details from the source.
       - DO NOT summarize or condense information
       - DO NOT skip details
       - Preserve specific values, timelines, percentages, drug names, and clinical criteria
    
    2.  **GRANULAR SUBTOPICS**: Break down the content into MANY focused subtopics.
       - Each subtopic should cover ONE specific concept (e.g., one disease process, one drug mechanism, one clinical syndrome)
       - Create as many subtopics as necessary to cover all material thoroughly
       - Aim for 5-10+ high-yield facts per subtopic for optimal mnemonic generation
    
    3.  **FACTUAL ACCURACY**: Stick to the ACTUAL content from the source material.
       - Quote specific facts, criteria, and details directly from the text
       - Include exact classifications, staging systems, diagnostic criteria
       - Preserve medical terminology and specific numerical values
    
    4.  **MNEMONIC-READY FORMAT**: Structure each subtopic to be ideal for mnemonic creation.
       - List key facts clearly (6-10 facts per subtopic is ideal)
       - Include associations, causes, symptoms, treatments separately
       - Use clear, memorable phrasing
    
    5.  **INCREMENTAL LEARNING**: Order subtopics logically.
       - Start with foundational concepts (anatomy, physiology, pathophysiology)
       - Progress to clinical presentation, diagnosis, and treatment
       - End with complications, prognosis, and special considerations

    Output Format (Strict Markdown):
    # [Comprehensive Descriptive Title Based on Source Material]
    
    ## [Subtopic 1: Specific Concept Name]
    **Overview**: [Brief context for this subtopic]
    
    **Key Facts** (for mnemonic creation):
    - [Fact 1 with specific details]
    - [Fact 2 with specific details]
    - [Fact 3 with specific details]
    - [... 5-10+ key facts from source]
    
    [Additional explanatory text if needed for context]

    ## [Subtopic 2: Next Specific Concept]
    **Overview**: [Brief context]
    
    **Key Facts**:
    - [Comprehensive list of facts from source]
    
    [Continue for ALL concepts in the source material...]
    
    REMEMBER: The goal is to preserve ALL important information from the source text and create as many subtopics as needed to ensure complete coverage for mnemonic generation. Don't hold back on details!
    """

def get_speech_prompt(text: str, language: str) -> str:
    lang_name = 'Spanish' if language == 'es' else 'English'
    return f"Read the following aloud in a warm, friendly and engaging tone ({lang_name}): {text}"

