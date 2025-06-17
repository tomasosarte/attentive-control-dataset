from typing import List
from pydantic import BaseModel, Field


class TransformationConfig(BaseModel):
    pass

class RotationConfig(TransformationConfig):
    angle: float = Field(..., description="Rotation angle in degrees")

class TranslationConfig(TransformationConfig):
    x: int = Field(..., description="Horizontal shift")
    y: int = Field(..., description="Vertical shift")

class Config(BaseModel):
    dataset: str
    data_dir: str
    output_dir: str
    transformations: List[TransformationConfig]
    proportions: List[float]
    include_original: bool