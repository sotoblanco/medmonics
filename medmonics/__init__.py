"""
MedMonics - AI-Powered Medical Mnemonics Generator

This package provides the core functionality for generating memorable medical mnemonics
using Google's Gemini AI. It includes:

- Pipeline: 5-step generation process (mnemonic → visual prompt → image → bboxes → quiz)
- Prompts: All prompt templates and model configurations
- Data Loader: Utilities for parsing batch API results
- Schemas: Pydantic models for type safety

Main exports:
    - MedMnemonicPipeline: Main orchestrator class for generation
    - MnemonicResponse: Data model for mnemonic results
    - BboxAnalysisResponse: Bounding box data model
    - QuizList, QuizItem: Quiz data models
    - Association: Character-to-medical-term mapping

Usage:
    from medmonics import MedMnemonicPipeline
    
    pipeline = MedMnemonicPipeline()
    result = pipeline.step1_generate_mnemonic(
        topic="Cushing's Syndrome",
        language="en",
        theme="Standard Mnemonic"
    )
"""

from .pipeline import (
    MedMnemonicPipeline,
    MnemonicResponse,
    BboxAnalysisResponse,
    QuizList,
    QuizItem,
    Association,
)

__all__ = [
    "MedMnemonicPipeline",
    "MnemonicResponse",
    "BboxAnalysisResponse",
    "QuizList",
    "QuizItem",
    "Association",
]

__version__ = "0.1.0"
