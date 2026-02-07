import streamlit as st
import os
import io
import json
import re
import random
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from medmonics.pipeline import MedMnemonicPipeline, MnemonicResponse, QuizList, BboxAnalysisResponse, Association
from medmonics.data_loader import parse_jsonl_results
from scripts import batch_submit, batch_retrieve
from medmonics.storage import LocalStorage, GCSBackend
import tomllib
from dotenv import load_dotenv


# Load variables
load_dotenv()

# Initialize Storage
@st.cache_resource
def get_storage():
    secrets = {}
    
    # 1. Try Streamlit secrets (Cloud / .streamlit/secrets.toml)
    try:
        if hasattr(st, "secrets") and "gcp_service_account" in st.secrets:
            secrets = st.secrets
    except Exception:
        # Secrets file not found, will fall back to local secrets.toml
        pass
    
    # 2. Fallback: Try loading secrets.toml from root (Local Dev convenience)
    if "gcp_service_account" not in secrets and os.path.exists("secrets.toml"):
        try:
            with open("secrets.toml", "rb") as f:
                secrets = tomllib.load(f)
        except Exception as e:
            st.warning(f"Found secrets.toml but failed to load: {e}")

    # Initialize GCS if secrets found
    if "gcp_service_account" in secrets:
        try:
            return GCSBackend(
                bucket_name=secrets["general"]["bucket_name"],
                service_account_info=secrets["gcp_service_account"]
            )
        except Exception as e:
            st.error(f"Failed to initialize GCS Backend: {e}")
            return LocalStorage()
            
    return LocalStorage()

storage_backend = get_storage()

# Define storage path for local operations (Batch)
STORAGE_DIR = Path("generations")
STORAGE_DIR.mkdir(exist_ok=True)

# Medical Specialties
SPECIALTIES = [
    "General",
    "Internal Medicine",
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
    "Nephrology",
    "Endocrinology",
    "Hematology",
    "Infectious Disease",
    "Rheumatology",
    "Neurology",
    "Psychiatry",
    "Dermatology",
    "Ophthalmology",
    "Otorhinolaryngology",
    "Pediatrics",
    "Obstetrics & Gynecology",
    "Surgery",
    "Orthopedics",
    "Urology",
    "Oncology",
    "Emergency Medicine",
    "Pharmacology",
    "Pathology",
    "Radiology",
    "Anatomy",
    "Physiology",
    "Microbiology",
    "Biochemistry",
]

def slugify(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_-]+', '_', text)
    return text[:30]

def save_generation(mnemonic_data, bbox_data, quiz_data, image_bytes, specialty="General", parent_id=None):
    return storage_backend.save_generation(
        mnemonic_data, bbox_data, quiz_data, image_bytes, specialty, parent_id
    )

def list_generations(specialty_filter=None):
    return storage_backend.list_generations(specialty_filter)

def load_generation(folder_path):
    return storage_backend.load_generation(folder_path)

@st.cache_data(ttl=60)
def get_all_challenge_items():
    challenge_pool = []
    try:
        folders = list_generations()
    except Exception as e:
         # st.error(f"Error listing generations: {e}")
         return []

    for folder in folders:
        try:
            # folder is a dict with 'identifier'
            m_data, b_data, q_data, i_bytes, meta = load_generation(folder['identifier'])
            
            # Helper to handle dict vs object mismatch during transition
            if isinstance(q_data, dict):
                 q_data = QuizList(**q_data)
            if isinstance(m_data, dict):
                 m_data = MnemonicResponse(**m_data)
            if isinstance(b_data, dict):
                 b_data = BboxAnalysisResponse(**b_data)

            if q_data and q_data.quizzes:
                for quiz in q_data.quizzes:
                    challenge_pool.append({
                        "quiz": quiz,
                        "image_bytes": i_bytes,
                        "bbox_data": b_data,
                        "topic": m_data.topic,
                        "mnemonic_data": m_data.model_dump()
                    })
        except Exception as e:
            # Skip corrupted folders
            # st.warning(f"Error loading {folder.get('name', 'unknown')}: {e}")
            continue
    return challenge_pool

# Page config
st.set_page_config(
    page_title="MedMnemonic AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more premium look
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    .main-title {
        font-size: 3rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(#eee, #333);
        background: linear-gradient(45deg, #ff00cc, #3333ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        background: linear-gradient(45deg, #7b2ff7, #2196f3);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .association-item {
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        border-left: 4px solid #7b2ff7;
        background: rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to draw bboxes
def draw_bboxes(image_bytes, bbox_data: BboxAnalysisResponse, focus_character: str = None):
    img_color = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img_color.size
    
    # Use a generic font
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = ["#FF3D00", "#2979FF", "#00E676", "#FFEA00", "#D500F9", "#00B0FF"]
    
    if focus_character:
        # Create grayscale version of the image
        img_gray = img_color.convert("L").convert("RGBA")
        # Start with the gray image as background
        base_img = img_gray
        draw = ImageDraw.Draw(base_img)
        
        items = bbox_data.boxes if bbox_data.boxes else []
        for i, item in enumerate(items):
            if item.character and focus_character and item.character.strip().lower() == focus_character.strip().lower():
                box = item.box_2d
                if not box or all(v == 0 for v in box) or len(box) < 4:
                    continue
                
                ymin, xmin, ymax, xmax = box
                left, top, right, bottom = xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000
                
                # Paste the color version of the character back onto the gray background
                character_crop = img_color.crop((left, top, right, bottom))
                base_img.paste(character_crop, (int(left), int(top)))
                
                # Draw the colored box
                color = colors[i % len(colors)]
                draw.rectangle([left, top, right, bottom], outline=color, width=8)
                draw.text((left + 5, top - 30), item.character, fill=color, font=font)
                break # Only highlight the first match for that character
        
        return base_img.convert("RGB")
    else:
        # Standard view: all colored with all boxes
        draw = ImageDraw.Draw(img_color)
        items = bbox_data.boxes if bbox_data.boxes else []
        for i, item in enumerate(items):
            box = item.box_2d
            if not box or all(v == 0 for v in box) or len(box) < 4:
                continue
            
            ymin, xmin, ymax, xmax = box
            left, top, right, bottom = xmin * w / 1000, ymin * h / 1000, xmax * w / 1000, ymax * h / 1000
            
            color = colors[i % len(colors)]
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            draw.text((left + 5, top + 5), item.character, fill=color, font=font)
        
        return img_color.convert("RGB")

# Initialize Pipeline
@st.cache_resource
def get_pipeline():
    return MedMnemonicPipeline()

pipeline = get_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/200/000000/brain.png", width=150)
    st.header("Settings")
    specialty = st.selectbox("🩺 Medical Specialty", SPECIALTIES)
    language = st.selectbox("Language", ["en", "es"], format_func=lambda x: "English" if x == "en" else "Español")
    visual_style = st.selectbox("🎨 Visual Style", ["Cartoon", "Photorealistic", "Professional"], index=0).lower()
    theme = st.text_input("Theme", placeholder="e.g. Cyberpunk, Medieval, Pixar")
    if not theme:
        theme = "Professional Educator"
    
    st.divider()
    st.divider()
    
    # Auto-refreshing history component
    @st.fragment(run_every=30)
    def render_history_sidebar():
        st.markdown("### 🏺 History (Live)")
        
        # 1. Fetch ALL generations first to find available folders
        all_gens = list_generations(None)
        
        # 2. Extract unique folder names (categories) from identifiers
        # Identifier format expected: "Folder/Timestamp_Topic"
        available_folders = set()
        for g in all_gens:
            parts = g['identifier'].split('/')
            if len(parts) > 1:
                available_folders.add(parts[0])
            else:
                available_folders.add("Uncategorized")
        
        sorted_folders = sorted(list(available_folders))
        
        # 3. Dynamic Filter
        history_filter = st.selectbox("Filter by Folder:", ["All"] + sorted_folders)
        
        # 4. Filter the list based on selection
        if history_filter == "All":
            previous_gens = all_gens
        else:
            previous_gens = [g for g in all_gens if g['identifier'].startswith(f"{history_filter}/")]
        
        if previous_gens:
            # previous_gens is list of dicts: {'name': ..., 'identifier': ...}
            gen_names = {g['name']: g['identifier'] for g in previous_gens}
            selected_gen_name = st.selectbox("Load previous mnemonic:", 
                                             ["-- Select --"] + list(gen_names.keys()))
            
            if selected_gen_name != "-- Select --":
                if st.button("Load Selected"):
                    folder_id = gen_names[selected_gen_name]
                    try:
                        m_data, b_data, q_data, i_bytes, meta = load_generation(folder_id)
                        
                        # Convert dicts back to Pydantic models if they are dicts
                        if isinstance(m_data, dict):
                            m_data = MnemonicResponse(**m_data)
                        if isinstance(b_data, dict):
                            b_data = BboxAnalysisResponse(**b_data)
                        if isinstance(q_data, dict):
                            q_data = QuizList(**q_data)
                            
                        st.session_state['mnemonic_data'] = m_data
                        st.session_state['bbox_data'] = b_data
                        st.session_state['quiz_data'] = q_data
                        st.session_state['image_bytes'] = i_bytes
                        # Store topic_id as current_generation_id for recursive linking
                        st.session_state['current_generation_id'] = meta.get('topic_id')
                        st.session_state['parent_id_for_save'] = meta.get('parent_id')
                        
                        st.success(f"Loaded: {selected_gen_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
        else:
            st.write("No saved generations found.")

    render_history_sidebar()

    st.info("MedMnemonic AI uses Gemini to store creations in the cloud.")

def run_generation_pipeline(topic, language, theme, visual_style="cartoon", specialty="General", parent_id=None):
    if 'last_autosave_path' in st.session_state:
        del st.session_state['last_autosave_path']
        
    with st.status(f"Executing MedMnemonic Pipeline for '{topic[:20]}...'...", expanded=True) as status:
        try:
            # Step 1: Mnemonic
            st.write("Step 1: Analyzing facts and creating story...")
            mnemonic_data = pipeline.step1_generate_mnemonic(topic, language, theme, visual_style)
            st.session_state['mnemonic_data'] = mnemonic_data
            
            # Step 2: Visual Enhancement
            st.write("Step 2: Enhancing visual prompt...")
            enhanced_v_prompt = pipeline.step2_enhance_visual_prompt(mnemonic_data, theme)
            st.session_state['enhanced_v_prompt'] = enhanced_v_prompt
            
            # Step 3: Image Generation
            st.write("Step 3: Generating mnemonic illustration...")
            image_bytes = pipeline.step3_generate_image(enhanced_v_prompt, theme, visual_style)
            st.session_state['image_bytes'] = image_bytes
            
            # Step 4: Bbox Analysis
            st.write("Step 4: Analyzing character locations...")
            bbox_data = pipeline.step4_analyze_bboxes(image_bytes, mnemonic_data)
            st.session_state['bbox_data'] = bbox_data
            
            # Step 5: Quiz Generation
            st.write("Step 5: Preparing challenge questions...")
            quiz_data = pipeline.step5_generate_quiz(mnemonic_data, language)
            st.session_state['quiz_data'] = quiz_data
            
            st.session_state['current_generation_id'] = None # Reset unless saved
            st.session_state['parent_id_for_save'] = parent_id # Store for save button
            
            # AUTOSAVE
            try:
                st.write("Step 6: Autosaving mnemonic...")
                saved_path_id = save_generation(
                    mnemonic_data,
                    bbox_data,
                    quiz_data,
                    image_bytes,
                    specialty=specialty,
                    parent_id=parent_id
                )
                # Store the ID/Name for display
                st.session_state['last_autosave_path'] = saved_path_id
                get_all_challenge_items.clear()
            except Exception as save_err:
                st.warning(f"Autosave failed: {save_err}")
            
            status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)
            return True
        except Exception as e:
            status.update(label="❌ Pipeline Failed!", state="error", expanded=True)
            st.error(f"Error during execution: {str(e)}")
            return False

# Main UI
st.markdown('<h1 class="main-title">🧠 MedMnemonic AI</h1>', unsafe_allow_html=True)
st.markdown("""
### How it works:
1. **Input**: Provide medical facts or a topic.
2. **Story**: AI creates a wacky mnemonic story with sound-alike characters.
3. **Visualize**: AI generates an illustration and identifies characters.
4. **Interact**: Click on facts to highlight characters and dim the rest of the image.
5. **Test**: Take the interactive quiz to reinforce your memory.
""")

# TOP LEVEL TABS
main_tabs = st.tabs(["✨ Generator", "🧠 Global Challenge", "🚀 Batch Prep"])

# --- TAB 1: GENERATOR (Original Flow) ---
with main_tabs[0]:
    topic_input = st.text_area("Enter Medical Topic / Facts", 
                               placeholder="e.g. Cushing's Syndrome symptoms: Moon face, buffalo hump, hypertension...",
                               height=100)

    if st.button("Generate Mnemonic Plan"):
        if not topic_input:
            st.warning("Please enter a topic first.")
        else:
            run_generation_pipeline(topic_input, language, theme, visual_style, specialty=specialty)

    # Display Results if Available
    if 'mnemonic_data' in st.session_state:
        data = st.session_state['mnemonic_data']
        
        # Save info / Autosave confirmation
        col_save1, col_save2 = st.columns([3, 1])
        with col_save2:
            if 'last_autosave_path' in st.session_state:
                st.success(f"💾 Autosaved: {st.session_state['last_autosave_path']}")
            else:
                if st.button("💾 Save this Mnemonic"):
                    try:
                        path_id = save_generation(
                            st.session_state['mnemonic_data'],
                            st.session_state['bbox_data'],
                            st.session_state['quiz_data'],
                            st.session_state['image_bytes'],
                            parent_id=st.session_state.get('parent_id_for_save'),
                            specialty=specialty
                        )
                        get_all_challenge_items.clear()
                        st.success(f"Saved to: {path_id}")
                    except Exception as e:
                        st.error(f"Error saving: {e}")

        st.divider()
        st.header("Interactive Study Mode")

        # Key Facts Section
        st.subheader("📋 Key Medical Facts")
        with st.expander("Show High-Yield Facts", expanded=True):
            for fact in (data.facts or []):
                st.markdown(f"- {fact}")
        
        st.divider()
        
        # Story Section
        st.subheader("Mnemonic Story")
        st.markdown(f'<div class="card">{data.story}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1.5, 1])
        
        with col2:
            st.subheader("Associations & Facts")
            st.write("Click on a fact to highlight it in the illustration.")
            
            # Use a radio button to select character for highlighting
            options = ["Show All"] + [a.character for a in data.associations]
            selected_char = st.radio("Select Character to Focus:", options, label_visibility="collapsed")
            
            focus_name = None if selected_char == "Show All" else selected_char
            
            # Display the facts in a list synchronized with selection
            for assoc in data.associations:
                is_selected = (assoc.character == selected_char)
                border_color = "#ff00cc" if is_selected else "#7b2ff7"
                bg_color = "rgba(255, 0, 204, 0.1)" if is_selected else "rgba(255, 255, 255, 0.05)"
                
                st.markdown(f"""
                <div class="association-item" style="border-left: 6px solid {border_color}; background: {bg_color};">
                    <b style="color: {border_color if is_selected else 'white'}">{assoc.character}</b> ➔ <i>{assoc.medicalTerm}</i><br>
                    <small>{assoc.explanation}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Dive Deeper Button
                if st.button(f"🔍 Dive Deeper into {assoc.medicalTerm}", key=f"dive_{assoc.character}"):
                    st.toast(f"Starting Dive Deeper for {assoc.medicalTerm}...")
                    # Logic for recursive generation
                    new_topic = f"{assoc.medicalTerm}: {assoc.explanation} (Context: {data.topic})"
                    st.write(f"🔄 Triggering generation for: **{new_topic}**")
                    
                    
                    # We need the current generation folder name or ID to link it
                    parent_id = None
                    if 'current_generation_id' in st.session_state and st.session_state['current_generation_id']:
                         parent_id = st.session_state['current_generation_id']
                    # Use the current theme
                    current_theme = theme if theme else "Professional Educator"

                    if run_generation_pipeline(new_topic, language, current_theme, visual_style=visual_style, specialty=specialty, parent_id=parent_id):
                        pass
                    st.rerun()

        with col1:
            st.subheader("Mnemonic Illustration")
            if 'image_bytes' in st.session_state:
                if 'bbox_data' in st.session_state:
                    annotated_img = draw_bboxes(st.session_state['image_bytes'], st.session_state['bbox_data'], focus_character=focus_name)
                    st.image(annotated_img, width='stretch')
                else:
                    st.image(st.session_state['image_bytes'], width='stretch')
            else:
                st.info("Image will appear here once generated.")
        
        # Generator-specific Quiz Section
        st.divider()
        st.subheader("❓ Quiz")
        if 'quiz_data' in st.session_state:
            quiz_list = st.session_state['quiz_data']
            for i, q in enumerate(quiz_list.quizzes):
                with st.expander(f"Question {i+1}: {q.question[:50]}..."):
                    st.write(q.question)
                    choice = st.radio(f"Select an option for Q{i+1}", q.options, key=f"q_{i}")
                    if st.button(f"Check Answer for Q{i+1}"):
                        if q.options.index(choice) == q.correctOptionIndex:
                            st.success(f"Correct! {q.explanation}")
                        else:
                            st.error(f"Incorrect. The correct answer was: {q.options[q.correctOptionIndex]}. {q.explanation}")


# --- TAB 2: GLOBAL CHALLENGE ---
with main_tabs[1]:
    st.header("🧠 Global Visual Challenge")
    # Pre-load all stories: Collect questions from all saved folders
    challenge_pool = get_all_challenge_items()
    
    if challenge_pool:
        # Use session state to keep track of the current challenge item
        if 'pool_q_idx' not in st.session_state or st.session_state['pool_q_idx'] >= len(challenge_pool):
            st.session_state['pool_q_idx'] = random.randint(0, len(challenge_pool) - 1)
        
        if st.button("🎲 Next Random Challenge"):
            st.session_state['pool_q_idx'] = random.randint(0, len(challenge_pool) - 1)
            st.rerun()
        item = challenge_pool[st.session_state['pool_q_idx']]
        current_q = item["quiz"]

        st.markdown(f"**Current Topic Review:** {item['topic']}")
        
        v_col1, v_col2 = st.columns([1.5, 1])
        
        with v_col1:
            st.subheader("Illustration Hint")
            # We highlight ONLY the relevant character
            focus_char = getattr(current_q, 'character', None)
            hint_img = draw_bboxes(item["image_bytes"], item["bbox_data"], focus_character=focus_char)
            st.image(hint_img, width='stretch')

        with v_col2:
            st.subheader("Challenge")
            st.write(current_q.question)
            
            # Use a unique key based on the pool index and question to avoid collisions
            q_key = f"pool_radio_{st.session_state['pool_q_idx']}"
            choice = st.radio("Choose:", current_q.options, key=q_key, label_visibility="collapsed")
            
            if st.button("Submit", type="primary", key=f"pool_submit_{st.session_state['pool_q_idx']}"):
                if current_q.options.index(choice) == current_q.correctOptionIndex:
                    st.success(f"🎊 Correct!")
                else:
                    st.error(f"❌ Incorrect.")
                st.info(f"**Explanation:** {current_q.explanation}")
    else:
        st.warning("No saved mnemonics found. Please generate and save a mnemonic first to play the global challenge!")


# --- TAB 3: BATCH PREP ---
with main_tabs[2]:
    st.header("🚀 Batch Input Preparation")
    st.markdown("Use this tool to research tasks and prepare a batch of mnemonics to be generated in the background.")

    # 1. Input Method
    input_method = st.radio("Input Method:", ["Topic Research", "Content Upload"], horizontal=True)

    generated_markdown = None
    
    if input_method == "Topic Research":
        batch_topic = st.text_input("Enter High-Level Topic:", placeholder="e.g. Heart Failure, Diabetes Types")
        if st.button("🧪 Research & Breakdown Topic"):
            with st.spinner("Researching and breaking down topic..."):
                try:
                    markdown_res = pipeline.generate_breakdown_markdown(batch_topic, input_type="topic", language=language)
                    st.session_state['batch_markdown'] = markdown_res
                    st.session_state['batch_original_input'] = batch_topic
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    else: # Content Upload
        uploaded_file = st.file_uploader("Upload Medical Content (PDF/Text)", type=["pdf", "txt", "md"])
        if uploaded_file and st.button("📄 Analyze Content"):
            with st.spinner("Analyzing content..."):
                try:
                    bytes_data = uploaded_file.getvalue()
                    markdown_res = pipeline.generate_breakdown_markdown(bytes_data, input_type="content", language=language)
                    st.session_state['batch_markdown'] = markdown_res
                    st.session_state['batch_original_input'] = uploaded_file.name
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    # 2. Display & Edit
    if 'batch_markdown' in st.session_state:
        st.subheader("Review Breakdown")
        st.info("The following subtopics were identified. Each '## Header' will become a separate mnemonic generation task.")
        edited_markdown = st.text_area("Edit Markdown if needed:", value=st.session_state['batch_markdown'], height=400)
        st.session_state['batch_markdown'] = edited_markdown # Update state on edit potentially? 

        # 3. Parsed Preview
        original_input = st.session_state.get('batch_original_input', "Unknown Topic")
        items = pipeline.parse_markdown_to_items(edited_markdown, language, original_input=original_input, visual_style=visual_style)
        
        # Display Research Header (First line starting with #)
        first_line = edited_markdown.strip().split('\n')[0]
        research_title = first_line[2:].strip() if first_line.startswith("# ") else "Research Breakdown"
        st.write(f"### Research: {research_title}")

        st.write(f"**Found {len(items)} subtopics:**")
        st.dataframe(items)

        # 4. Approve & Submit
        col_b1, col_b2, col_b3 = st.columns([1,1,1])
        with col_b1:
            if st.button("🚀 Process & Submit Batch", type="primary"):
                try:
                    # 1. Immediate Text Generation Phase
                    staging_items = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, item in enumerate(items):
                        status_text.text(f"Processing Text ({i+1}/{len(items)}): {item['title']}...")
                        
                        # Step 1: Mnemonic
                        m_data = pipeline.step1_generate_mnemonic(item["topic"], item["language"], theme, item["visual_style"])
                        # Step 2: Visual Enhancement
                        v_prompt = pipeline.step2_enhance_visual_prompt(m_data, theme)
                        # Step 5: Quiz
                        q_data = pipeline.step5_generate_quiz(m_data, item["language"])
                        
                        # Package for staging
                        staged_item = {
                            "input": item["input"],
                            "title": item["title"],
                            "language": item["language"],
                            "visual_style": item["visual_style"],
                            "theme": theme,
                            "mnemonic_data": m_data.model_dump(),
                            "visual_prompt": v_prompt,
                            "quiz_data": q_data.model_dump()
                        }
                        staging_items.append(staged_item)
                        progress_bar.progress((i + 1) / len(items))

                    # 2. Save to Staging File
                    with open(batch_submit.STAGING_FILE, 'w', encoding='utf-8') as f:
                        json.dump(staging_items, f, indent=4, ensure_ascii=False)
                    
                    st.success(f"Generated text for {len(staging_items)} items!")
                    
                    # 3. Submit Batch Job for Images
                    with st.status("🚀 Submitting Image Batch Job...", expanded=True) as status:
                        job_id = batch_submit.submit_batch_job(batch_submit.STAGING_FILE)
                        if job_id:
                            status.update(label=f"✅ Image Job Submitted! ID: {job_id}", state="complete")
                            st.balloons()
                        else:
                            status.update(label="❌ Image Job Submission Failed", state="error")
                            
                except Exception as e:
                    st.error(f"Failed to process: {e}")

        with col_b3:
            if st.button("🗑️ Clear Staged Markdown"):
                del st.session_state['batch_markdown']
                st.rerun()


    
    st.divider()
    st.subheader("📊 Batch Job Status")
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        if st.button("🔄 Check Job Status"):
            with st.spinner("Checking..."):
                status = batch_retrieve.check_batch_status()
                st.session_state['batch_status'] = status
    
    if 'batch_status' in st.session_state:
        status = st.session_state['batch_status']
        st.info(f"**State:** {status.get('state')} | **Job Name:** {status.get('job_name', 'N/A')}")
        st.write(f"**Message:** {status.get('message')}")
        
        if status.get('state') == 'JOB_STATE_SUCCEEDED':
            st.success(f"Job completed! Ready to retrieve results.")
            
            if st.button("☁️ Retrieve & Save to Cloud"):
                with st.spinner("Retrieving results and saving to Storage..."):
                    try:
                        count = batch_retrieve.retrieve_and_finalize(status.get('job_name'), storage_backend=storage_backend)
                        if count > 0:
                            st.balloons()
                            st.success(f"✅ Successfully saved {count} mnemonics to Cloud Storage!")
                            st.info("Check the 'History' sidebar to see your new mnemonics.")
                        else:
                            st.warning("No items were processed. Check logs or job status.")
                    except Exception as e:
                        st.error(f"Error during retrieval: {e}")
