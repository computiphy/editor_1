"""
Album Layout Engine (Stage 9)
=============================
Orchestrates template-based album generation:
  1. Discover images from final/ (and cutouts/ if enabled)
  2. Chunk images into pages based on template image counts
  3. Select backgrounds via LAB ΔE color matching
  4. Render full-resolution JPEG spreads
  5. Save project.json for rebuild capability
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from src.layout.models import (
    CellPlacement, PageLayout, AlbumProject,
)
from src.layout.template_registry import TemplateRegistry
from src.layout.background_selector import BackgroundSelector
from src.layout.renderer import AlbumRenderer


class AlbumLayoutEngine:
    """
    Main entry point for Stage 9.
    Generates album pages and renders them to JPEG.
    """

    def __init__(self,
                 mode: str = "template",
                 page_size: Tuple[int, int] = (3600, 2400),
                 dpi: int = 300,
                 images_per_page: int = 0,
                 padding: int = 60,
                 gutter: int = 30,
                 use_cutouts: bool = False,
                 background_dir: str = "assets/backgrounds",
                 background_strategy: str = "dominant",
                 export_format: str = "jpeg",
                 export_quality: int = 95):

        self.mode = mode
        self.page_width, self.page_height = page_size
        self.dpi = dpi
        self.images_per_page = images_per_page
        self.padding = padding
        self.gutter = gutter
        self.use_cutouts = use_cutouts

        # Initialize submodules
        self.registry = TemplateRegistry()
        self.bg_selector = BackgroundSelector(
            Path(background_dir), strategy=background_strategy
        )
        self.renderer = AlbumRenderer(
            page_width=self.page_width,
            page_height=self.page_height,
            quality=export_quality,
            output_format=export_format,
        )

    def generate_album(self,
                       final_dir: Path,
                       cutouts_dir: Optional[Path],
                       output_dir: Path,
                       config_snapshot: Optional[dict] = None) -> AlbumProject:
        """
        Generate a complete album from pipeline output.

        Args:
            final_dir: Path to final/ directory with graded images.
            cutouts_dir: Path to cutouts/ directory (optional).
            output_dir: Where to write album pages and project.json.
            config_snapshot: Frozen config dict for project state.

        Returns:
            AlbumProject with all page definitions.
        """
        # 1. Discover images
        images = self._discover_images(final_dir)
        if not images:
            print("Warning: No images found in final/ — skipping album generation")
            return AlbumProject(
                config_snapshot=config_snapshot or {},
                pages=[], total_images=0,
            )

        print(f"    Found {len(images)} images for album")

        # Build cutout lookup
        cutout_map = {}
        if self.use_cutouts and cutouts_dir and cutouts_dir.exists():
            for f in cutouts_dir.glob("*.png"):
                cutout_map[f.stem] = f
            print(f"    Found {len(cutout_map)} cutouts")

        # 2. Determine images per page
        ipp = self._compute_images_per_page(len(images))
        print(f"    Images per page: {ipp} (mode: {self.mode})")

        # 3. Chunk images into pages
        pages = []
        image_chunks = self._chunk_images(images, ipp)

        album_dir = output_dir / "album"
        album_dir.mkdir(parents=True, exist_ok=True)

        # 4. Generate page layouts + render
        print(f"--- Stage 9: Album Layout ({len(image_chunks)} pages) ---")
        for page_idx, chunk in enumerate(tqdm(image_chunks, desc="Rendering pages")):
            page = self._generate_page(
                page_number=page_idx + 1,
                image_paths=chunk,
                cutout_map=cutout_map,
            )
            pages.append(page)

            # Render to JPEG
            page_file = album_dir / f"page_{page.page_number:03d}.jpg"
            self.renderer.render_page(page, page_file, use_cutouts=self.use_cutouts)

        # 5. Save project state
        project = AlbumProject(
            config_snapshot=config_snapshot or {},
            pages=pages,
            total_images=len(images),
        )
        project.save(album_dir / "project.json")
        print(f"    Album saved: {album_dir} ({len(pages)} pages)")

        return project

    def _discover_images(self, final_dir: Path) -> List[Path]:
        """Discover and sort images from final/ directory."""
        if not final_dir.exists():
            return []

        extensions = {".jpg", ".jpeg", ".png"}
        images = sorted([
            f for f in final_dir.iterdir()
            if f.suffix.lower() in extensions
        ])
        return images

    def _compute_images_per_page(self, total: int) -> int:
        """Determine images per page using the heuristic from FRD §5.3."""
        if self.images_per_page > 0:
            return self.images_per_page

        # Auto mode
        if total <= 6:
            return max(1, min(2, total))
        elif total <= 20:
            return 3
        elif total <= 50:
            return 4
        else:
            return 5

    def _chunk_images(self, images: List[Path], ipp: int) -> List[List[Path]]:
        """Split images into page-sized chunks."""
        chunks = []
        for i in range(0, len(images), ipp):
            chunk = images[i:i + ipp]
            chunks.append(chunk)
        return chunks

    def _generate_page(self, page_number: int, image_paths: List[Path],
                       cutout_map: Dict[str, Path]) -> PageLayout:
        """
        Generate a PageLayout for a set of images.
        Selects template, resolves cell coordinates, picks background.
        """
        count = len(image_paths)

        # Select template
        template = self.registry.get_for_count(count)
        if template is None:
            # Fallback: simple grid
            template = self.registry.get_for_count(1)

        # Compute usable area (inside padding)
        usable_w = self.page_width - 2 * self.padding
        usable_h = self.page_height - 2 * self.padding

        # Convert normalized cells to pixel coordinates
        cells = []
        template_cells = template.cells[:count]  # Only use as many cells as images

        for i, (img_path, tcell) in enumerate(zip(image_paths, template_cells)):
            # Account for gutters between cells
            cell_x = self.padding + int(tcell.x * usable_w) + (self.gutter if tcell.x > 0.01 else 0)
            cell_y = self.padding + int(tcell.y * usable_h) + (self.gutter if tcell.y > 0.01 else 0)
            cell_w = int(tcell.w * usable_w) - (self.gutter if tcell.x > 0.01 else 0)
            cell_h = int(tcell.h * usable_h) - (self.gutter if tcell.y > 0.01 else 0)

            # Ensure minimum cell size
            cell_w = max(cell_w, 100)
            cell_h = max(cell_h, 100)

            # Resolve cutout path
            cutout_path = cutout_map.get(img_path.stem)

            cells.append(CellPlacement(
                image_path=img_path,
                cutout_path=cutout_path,
                x=cell_x, y=cell_y,
                width=cell_w, height=cell_h,
                z_order=i,
            ))

        # Select background
        bg_path = self.bg_selector.select(image_paths)

        return PageLayout(
            page_number=page_number,
            cells=cells,
            background_path=bg_path,
            layout_mode=self.mode,
            template_name=template.name if template else None,
        )
