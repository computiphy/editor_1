"""
Album Renderer
================
Composites full-resolution JPEG album spreads from PageLayout definitions.
CPU-bound via Pillow — no VRAM required.
"""

from pathlib import Path
from typing import Optional
from PIL import Image
from src.layout.models import PageLayout, CellPlacement


class AlbumRenderer:
    """
    Renders album pages to high-quality JPEG files.
    Processes one page at a time for memory efficiency.
    """

    def __init__(self, page_width: int = 3600, page_height: int = 2400,
                 quality: int = 95, output_format: str = "jpeg"):
        self.page_width = page_width
        self.page_height = page_height
        self.quality = quality
        self.output_format = output_format

    def render_page(self, page: PageLayout, output_path: Path,
                    use_cutouts: bool = False) -> Path:
        """
        Render a single album page to a JPEG file.

        Args:
            page: PageLayout with cell placements.
            output_path: Where to save the rendered JPEG.
            use_cutouts: If True, composite RGBA cutouts over background.

        Returns:
            Path to the rendered file.
        """
        # 1. Create canvas
        canvas = Image.new("RGB", (self.page_width, self.page_height),
                           color=page.background_color)

        # 2. Paste background image if available
        if page.background_path and page.background_path.exists():
            try:
                bg = Image.open(page.background_path).convert("RGB")
                bg = bg.resize((self.page_width, self.page_height), Image.LANCZOS)
                canvas.paste(bg)
                bg.close()
            except Exception as e:
                print(f"Warning: Could not load background {page.background_path}: {e}")

        # 3. Place each image (sorted by z_order)
        sorted_cells = sorted(page.cells, key=lambda c: c.z_order)
        for cell in sorted_cells:
            self._place_image(canvas, cell, use_cutouts)

        # 4. Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(output_path), "JPEG", quality=self.quality)
        canvas.close()

        return output_path

    def _place_image(self, canvas: Image.Image, cell: CellPlacement,
                     use_cutouts: bool):
        """Place a single image into its cell on the canvas."""
        # Determine source: cutout (RGBA) or flat image (RGB)
        source_path = None
        is_rgba = False

        if use_cutouts and cell.cutout_path and cell.cutout_path.exists():
            source_path = cell.cutout_path
            is_rgba = True
        elif cell.image_path.exists():
            source_path = cell.image_path
        else:
            print(f"Warning: Image not found: {cell.image_path}")
            return

        try:
            img = Image.open(source_path)

            if is_rgba:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")

            # Resize to fit cell, maintaining aspect ratio
            img = self._fit_to_cell(img, cell.width, cell.height)

            # Center within cell
            offset_x = cell.x + (cell.width - img.width) // 2
            offset_y = cell.y + (cell.height - img.height) // 2

            if is_rgba:
                # Alpha composite: need RGBA canvas temporarily
                temp = canvas.convert("RGBA")
                # Create a transparent layer with the image at position
                layer = Image.new("RGBA", temp.size, (0, 0, 0, 0))
                layer.paste(img, (offset_x, offset_y))
                temp = Image.alpha_composite(temp, layer)
                # Convert back to RGB
                result = temp.convert("RGB")
                canvas.paste(result)
                temp.close()
                layer.close()
                result.close()
            else:
                canvas.paste(img, (offset_x, offset_y))

            img.close()
        except Exception as e:
            print(f"Warning: Could not render {source_path}: {e}")

    def _fit_to_cell(self, img: Image.Image, cell_w: int, cell_h: int) -> Image.Image:
        """
        Resize image to fit within cell dimensions, preserving aspect ratio.
        Uses cover fit (fills the cell, may crop edges).
        """
        img_w, img_h = img.size
        # Calculate scale to cover the cell
        scale = max(cell_w / img_w, cell_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop to exact cell size
        left = (new_w - cell_w) // 2
        top = (new_h - cell_h) // 2
        img = img.crop((left, top, left + cell_w, top + cell_h))

        return img
