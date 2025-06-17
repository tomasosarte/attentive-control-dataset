from typing import List
from pydantic import BaseModel, Field


class Transformation(BaseModel):
    pass

class Rotation(Transformation):
    angle: float = Field(..., description="Rotation angle in degrees")

class Translation(Transformation):
    x: int = Field(..., description="Horizontal shift")
    y: int = Field(..., description="Vertical shift")

class Config(BaseModel):
    dataset: str
    data_dir: str
    output_dir: str
    transformations: List[Transformation]
    proportions: List[float]