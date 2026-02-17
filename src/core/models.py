from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

@dataclass(frozen=True)
class ImageScore:
    """Quality assessment result for a single image."""
    path: Path
    blur_score: float
    fft_energy: float
    brisque_score: float
    niqe_score: float
    has_faces: bool
    blink_detected: bool
    expression_score: float
    overall_quality: float
    passed: bool
    rejection_reasons: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class Cluster:
    """Group of near-duplicate images."""
    cluster_id: str
    images: List[Path]
    representative: Path
    hash_distances: Dict[str, int] = field(default_factory=dict)

@dataclass
class Chapter:
    """Narrative chapter grouping."""
    name: str
    images: List[Path]
    time_range: Optional[Tuple[str, str]] = None
    caption: Optional[str] = None
    confidence: float = 1.0

@dataclass
class LayoutPage:
    """Single album page layout specification."""
    page_number: int
    algorithm: str
    cells: List[Dict]
    aspect_ratio: Tuple[int, int] = (3, 2)

@dataclass
class PipelineResult:
    """Final pipeline run summary."""
    total_input: int
    total_culled: int
    total_restored: int
    total_graded: int
    album_pages: int
    elapsed_seconds: float
    errors: List[dict] = field(default_factory=list)
    checkpoint_path: Optional[Path] = None
