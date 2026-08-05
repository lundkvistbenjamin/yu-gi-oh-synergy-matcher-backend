from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class CardPredictionRequest(BaseModel):
    type: str
    race: str
    attribute: str
    atk: Optional[int] = -1
    def_val: Optional[int] = Field(default=-1, alias="def")
    defense: Optional[int] = None
    level: Optional[int] = 0

    @field_validator("atk", "def_val", "defense", "level", mode="before")
    @classmethod
    def sanitize_int(cls, value):
        if value is None or str(value).strip() == "":
            return -1
        try:
            parsed = int(float(value))
            return parsed if 0 <= parsed <= 99999 else -1
        except (ValueError, TypeError):
            return -1

    def get_defense(self) -> int:
        # Accommodates both 'def' and 'defense' keys without breaking frontend contracts
        if self.defense is not None and self.defense != -1:
            return self.defense
        return self.def_val if self.def_val is not None else -1

class PredictionItem(BaseModel):
    archetype: str
    confidence: float

class PredictionResponse(BaseModel):
    prediction: str
    top_predictions: List[PredictionItem]