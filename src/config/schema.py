from pydantic import BaseModel, Field
from typing import List, Optional

class PipelineConfig(BaseModel):
    name: str = "Wedding_Pipeline"
    input_dir: str
    input_formats: List[str] = Field(default_factory=lambda: ["raw", "jpg"])
    output_base: str
    gpu_backend: str = "auto"
    workers: int = 4
    metadata_required: bool = False

class CullingConfig(BaseModel):
    enabled: bool = True
    blur_threshold: float = 100.0
    duplicate_threshold: int = 5
    quality_threshold: float = 70.0

class TilingConfig(BaseModel):
    enabled: bool = True
    tile_size: int = 512
    overlap: int = 64

class RestorationConfig(BaseModel):
    enabled: bool = True
    auto_route: bool = True
    primary_model: str = "nafnet"
    face_restore: bool = True
    tiling: TilingConfig = Field(default_factory=TilingConfig)

class CroppingConfig(BaseModel):
    enabled: bool = True
    ratios: List[str] = Field(default_factory=lambda: ["1:1", "9:16"])
    detector: str = "rtdetr"

class WatermarkConfig(BaseModel):
    enabled: bool = True
    path: str = "assets/watermark.png"
    position: str = "auto"
    opacity: float = 0.5

class NarrativeConfig(BaseModel):
    enabled: bool = True
    clustering_eps: float = 0.5

class ColorGradingConfig(BaseModel):
    enabled: bool = False
    method: str = "lab_statistical"
    style: str = "natural"
    reference_image: Optional[str] = None
    strength: float = 1.0
    segmentation_enabled: bool = False

class BackgroundRemovalConfig(BaseModel):
    enabled: bool = False
    model: str = "u2net"

class LayoutExportConfig(BaseModel):
    format: str = "jpeg"
    quality: int = 95

class LayoutConfig(BaseModel):
    enabled: bool = False
    mode: str = "template"                                          # template | algorithmic | ai | hybrid | mixed
    page_size: List[int] = Field(default_factory=lambda: [3600, 2400])
    dpi: int = 300
    images_per_page: int = 0                                        # 0 = auto
    padding: int = 60
    gutter: int = 30
    use_cutouts: bool = False
    background_directory: str = "assets/backgrounds"
    background_strategy: str = "dominant"                            # dominant | average
    ai_style: str = "classic"                                       # classic | elegant | minimal | dynamic | magazine
    ai_seed: int = 42
    export: LayoutExportConfig = Field(default_factory=LayoutExportConfig)

class CroppingConfig(BaseModel):
    enabled: bool = True
    ratios: List[str] = Field(default_factory=lambda: ["1:1", "9:16"])
    detector: str = "rtdetr"

class WatermarkConfig(BaseModel):
    enabled: bool = True
    path: str = "assets/watermark.png"
    position: str = "auto"
    opacity: float = 0.5

class NarrativeConfig(BaseModel):
    enabled: bool = True
    clustering_eps: float = 0.5

class ConfigSchema(BaseModel):
    pipeline: PipelineConfig
    culling: CullingConfig = Field(default_factory=CullingConfig)
    restoration: RestorationConfig = Field(default_factory=RestorationConfig)
    color_grading: ColorGradingConfig = Field(default_factory=ColorGradingConfig)
    background_removal: BackgroundRemovalConfig = Field(default_factory=BackgroundRemovalConfig)
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    cropping: CroppingConfig = Field(default_factory=CroppingConfig)
    watermark: WatermarkConfig = Field(default_factory=WatermarkConfig)
    narrative: NarrativeConfig = Field(default_factory=NarrativeConfig)
