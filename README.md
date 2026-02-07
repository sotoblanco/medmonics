# MedMonics 🧠

**AI-Powered Medical Mnemonics Generator**

MedMonics uses Google's Gemini AI to create memorable, visual mnemonics for medical education. Transform complex medical concepts into engaging stories with illustrated characters, making study material easier to remember and more fun to learn.

## Features

- **5-Step AI Pipeline**: Automatic generation of mnemonics, visual prompts, images, bounding boxes, and quizzes
- **Multi-Language Support**: Generate content in English or Spanish
- **Batch Processing**: Generate multiple mnemonics efficiently using Gemini's Batch API
- **Interactive Quizzes**: Test your knowledge with AI-generated quiz questions
- **Visual Challenges**: Identify highlighted characters in grayscale images
- **Streamlit UI**: User-friendly web interface for all features

## Architecture

### Core Generation Pipeline (5 Steps)

1. **Generate Mnemonic** (`step1_generate_mnemonic`)
   - Creates story, associations, and initial visual prompt
   - Uses character-based puns for memory associations
   - Output: `MnemonicResponse` with topic, facts, story, associations, visualPrompt

2. **Enhance Visual Prompt** (`step2_enhance_visual_prompt`) 
   - Refines the visual description for better image generation
   - Adds artistic style and composition details
   - Output: Enhanced prompt string

3. **Generate Image** (`step3_generate_image`)
   - Creates illustration using Gemini's image generation
   - Falls back to simpler themes if generation fails
   - Output: Image bytes (PNG)

4. **Analyze Bounding Boxes** (`step4_analyze_bboxes`)
   - Identifies character locations in the generated image
   - Uses Gemini 1.5 Flash to detect 2D bounding boxes
   - Output: `BboxAnalysisResponse` with character positions

5. **Generate Quiz** (`step5_generate_quiz`)
   - Creates multiple-choice questions based on the mnemonic
   - Links each question to a specific character
   - Output: `QuizList` with questions, options, and explanations

### Batch Processing Workflow

1. **Batch Prep** (via Streamlit UI)
   - Input topic or upload content
   - AI breaks down into subtopics
   - Creates `batch_input.json` with breakdown items
   - Generates `batch_staging.json` with complete mnemonic data (steps 1-2, 5)

2. **Batch Submit** (`scripts/batch_submit.py`)
   - Reads `batch_staging.json`
   - Submits image generation requests (step 3) to Gemini Batch API
   - Saves job ID to `latest_batch_job.txt`

3. **Batch Retrieve** (`scripts/batch_retrieve.py`)
   - Checks job status
   - Downloads completed images from inline responses
   - Runs bbox analysis (step 4) on each image
   - Saves final results to `generations/` directory

## Installation

### Prerequisites
- Python 3.13+
- Google Gemini API key

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd medmonics

# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt

# Create .env file with your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Dependencies

- `google-genai` - Gemini API client
- `streamlit` - Web UI framework
- `pillow` - Image processing
- `pandas` - Data manipulation
- `python-dotenv` - Environment variable management
- `jupyter` - Optional, for notebooks

## Usage

### Running the Web App

```bash
streamlit run app.py
```

The app has 4 tabs:

1. **✨ Generator**: Create individual mnemonics
   - Enter a medical topic or facts
   - Select language, theme, and visual style
   - View generated story, image, and quiz
   - Save to local storage

2. **🧠 Global Challenge**: Practice with visual quizzes
   - Random quiz questions from all saved mnemonics
   - Character highlighting in images
   - Immediate feedback

3. **🚀 Batch Prep**: Prepare batch generation jobs
   - Input topic or upload content  
   - AI breaks down into subtopics
   - Review and edit breakdown
   - Submit batch job

4. **📂 Batch Results**: View and manage batch outputs
   - See all batch-generated mnemonics
   - Edit topic names
   - Save to permanent storage

### Command-Line Scripts

#### Submit a Batch Job

```bash
python scripts/batch_submit.py
```

Reads `data/batch_staging.json` and submits to Gemini Batch API.

#### Check Job Status

```bash
python scripts/batch_retrieve.py --status-only
```

#### Retrieve Completed Results

```bash
python scripts/batch_retrieve.py
```

Downloads images, runs bbox analysis, and saves to `generations/`.

#### Use Specific Job ID

```bash
python scripts/batch_retrieve.py --job-name "projects/12345/locations/us-central1/batchPredictionJobs/67890"
```

## File Structure

```
medmonics/
├── .env                      # API configuration (create this)
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Project dependencies
├── README.md               # This file
├── app.py                   # Main Streamlit application (911 lines)
│
├── medmonics/              # Core package
│   ├── __init__.py         # Package exports
│   ├── pipeline.py         # MedMnemonicPipeline class (260 lines)
│   ├── prompts.py          # Prompt templates and model constants (207 lines)
│   ├── data_loader.py      # Batch result parsing utilities (211 lines)
│   └── schemas.py          # Pydantic data models (12 lines)
│
├── scripts/                # Batch processing scripts
│   ├── __init__.py
│   ├── batch_submit.py     # Submit batch jobs (106 lines)
│   └── batch_retrieve.py   # Retrieve batch results (250 lines)
│
├── data/                   # Batch data files
│   ├── batch_input.json    # Original subtopic breakdown
│   ├── batch_staging.json  # Complete mnemonic data for batch
│   ├── batch_requests.jsonl # (historical, may not be used)
│   └── latest_batch_job.txt # Most recent batch job ID
│
└── generations/            # Saved mnemonics
    └── [specialty]/        # e.g., "cardiology", "neurology"
        └── [timestamp_topic]/
            ├── data.json   # Complete mnemonic data
            └── image.png   # Generated illustration
```

## Core Modules

### `medmonics/pipeline.py`
Main generation pipeline with `MedMnemonicPipeline` class containing all 5 generation steps plus batch breakdown methods.

**Key Classes:**
- `MedMnemonicPipeline` - Main orchestrator
- `MnemonicResponse` - Mnemonic data model
- `BboxAnalysisResponse` - Bounding box data
- `QuizList` - Quiz questions

### `medmonics/prompts.py`
All prompt templates and model configuration.

**Key Constants:**
- `MODEL_FLASH` - Fast model for text generation
- `MODEL_IMAGE_GEN` - Image generation model
- `MODEL_VISUAL_PROMPT` - Visual prompt enhancement

**Key Functions:**
- `get_mnemonic_prompt()` - Main mnemonic generation prompt
- `get_image_generation_prompt()` - Image generation prompt
- `get_bbox_analysis_prompt()` - Bbox detection prompt
- `get_quiz_prompt()` - Quiz generation prompt
- `get_topic_breakdown_prompt()` - Batch breakdown prompt

### `medmonics/data_loader.py`
Utilities for loading and normalizing batch API results.

**Key Functions:**
- `normalize_keys()` - Normalizes inconsistent JSON key formats
- `parse_jsonl_results()` - Parses batch output JSONL files

### `medmonics/schemas.py`
Pydantic models for type safety and validation.

## Development

### Adding New Prompt Templates

1. Add the template function to `medmonics/prompts.py`
2. Use the appropriate model constant
3. Include language instructions via `get_language_instruction()`

### Extending the Pipeline

To add a new generation step:

1. Add method to `MedMnemonicPipeline` class in `pipeline.py`
2. Update `run_generation_pipeline()` in `app.py`
3. Update data models in `schemas.py` if needed

### Modifying Visual Styles

Edit `get_visual_style_instruction()` in `prompts.py`:

```python
def get_visual_style_instruction(style: str):
    styles = {
        "cartoon": "Vibrant cartoon style...",
        "realistic": "Photorealistic style...",
        "your_style": "Your custom instructions..."
    }
    return styles.get(style, styles["cartoon"])
```

### Supported Languages

Currently: English (`en`) and Spanish (`es`)

To add a language, update `LANGUAGE_INSTRUCTION_*` constants and `get_language_instruction()` in `prompts.py`.

## Medical Specialties

The app supports saving to these specialty categories:

- General Medicine
- Cardiology
- Neurology
- Pediatrics
- Psychiatry
- Dermatology
- Gastroenterology
- Pulmonology
- Endocrinology
- Nephrology
- Immunology
- Infectious Diseases
- Obstetrics & Gynecology
- Surgery
- Orthopedics
- Urology
- Oncology
- Emergency Medicine
- Pharmacology
- Pathology
- Radiology
- Anatomy
- Physiology
- Microbiology
- Biochemistry

## Troubleshooting

### Batch Job Submission Fails

**Error:** `400 INVALID_ARGUMENT`
- Check that `batch_staging.json` exists and is valid JSON
- Verify API key in `.env` file
- Ensure `response_modalities` includes `'IMAGE'` for image generation

### No Images in Batch Results

**Symptom:** Batch completes but no images retrieved
- Verify job was submitted with inline requests (check `batch_submit.py`)
- Use `--status-only` flag to check job state
- Check that `response_modalities: ['TEXT', 'IMAGE']` was set

### Import Errors

```
ModuleNotFoundError: No module named 'medmonics'
```

Make sure you're running from the project root and the package is installed:
```bash
cd /path/to/medmonics
pip install -e .
```

### Pydantic Validation Errors

If you see validation errors when loading batch results, the JSON structure may have changed. Check `data_loader.py` and update `normalize_keys()` to handle the new format.

## License

[Your License Here]

## Contributing

[Your Contribution Guidelines Here]

## Contact

[Your Contact Information Here]
