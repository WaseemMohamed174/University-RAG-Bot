from pydantic import BaseModel


# ============== مخرجات Gemini ==============
class FinalAnswer(BaseModel):
    answer: str
    location_ids: list[str]


class Note(BaseModel):
    fact: str
    evidence: str
    source: str
    location_ids: list[str]


class BatchNotes(BaseModel):
    notes: list[Note]
