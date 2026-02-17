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

class ColorStyle(Enum):
    NATURAL = "natural"
    CINEMATIC = "cinematic"
    PASTEL = "pastel"
    MOODY = "moody"
    GOLDEN_HOUR = "golden_hour"
    FILM_KODAK = "film_kodak"
    FILM_FUJI = "film_fuji"
    VIBRANT = "vibrant"
    BLACK_AND_WHITE = "black_and_white"
    # New Presets
    MOODY_FOREST = "moody_forest"
    GOLDEN_HOUR_PORTRAIT = "golden_hour_portrait"
    URBAN_CYBERPUNK = "urban_cyberpunk"
    VINTAGE_PAINTERLY = "vintage_painterly"
    HIGH_FASHION = "high_fashion"
    SEPIA_MONOCHROME = "sepia_monochrome"
    VIBRANT_LANDSCAPE = "vibrant_landscape"
    LAVENDER_DREAM = "lavender_dream"
    BLEACH_BYPASS = "bleach_bypass"
    DARK_ACADEMIC = "dark_academic"

class CropRatio(Enum):
    SQUARE = "1:1"
    PORTRAIT = "4:5"
    STORY = "9:16"
    LANDSCAPE = "16:9"
