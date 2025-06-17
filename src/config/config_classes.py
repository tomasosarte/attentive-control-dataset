import yaml
from typing import List, Union, Literal, Annotated
from pydantic import BaseModel, Field

class TransformationConfig(BaseModel):
    type: str

class RotationConfig(TransformationConfig):
    type: Literal["rotation"]
    angle: float = Field(..., description="Rotation angle in degrees")

class TranslationConfig(TransformationConfig):
    type: Literal["translation"]
    x: int = Field(..., description="Horizontal shift")
    y: int = Field(..., description="Vertical shift")

TransformationUnion = Annotated[
    Union[RotationConfig, TranslationConfig],
    Field(discriminator="type")
]
class Config(BaseModel):
    dataset: str
    data_dir: str
    output_dir: str
    transformations: List[TransformationUnion]
    proportions: List[float]
    include_original: bool

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
