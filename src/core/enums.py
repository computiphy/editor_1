from enum import Enum

class QualityTier(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"

class RestorationMode(Enum):
    AUTO = "auto"
    DENOISE = "denoise"
    DEBLUR = "deblur"
    FACE = "face"
    ENHANCE = "enhance"

class LayoutAlgorithm(Enum):
    FIXED_PARTITION = "fixed_partition"
    FIXED_COLUMNS = "fixed_columns"
    HERO = "hero"
    DYNAMIC_COLLAGE = "dynamic_collage"
    MIXED = "mixed"

class ColorMethod(Enum):
    LAB_STATISTICAL = "lab_statistical"
    LUT = "lut"
    HISTOGRAM = "histogram"

class CropRatio(Enum):
    SQUARE = "1:1"
    PORTRAIT = "4:5"
    STORY = "9:16"
    LANDSCAPE = "16:9"
