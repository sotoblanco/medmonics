from pydantic import BaseModel
from typing import List, Optional
from .pipeline import Association, QuizItem, MnemonicResponse

class BatchItemResponse(BaseModel):
    topic: str
    facts: List[str]
    story: str
    associations: List[Association]
    visualPrompt: str
    quizzes: List[QuizItem]
