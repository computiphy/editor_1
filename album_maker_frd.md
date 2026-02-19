# Feature Requirements Document (FRD)
## Album Layout Module — Integration with Wedding Photography AI Pipeline

---

# 1. Purpose

This document defines the upgraded requirements for the **Album Layout module (Stage 9)** within the existing Wedding Photography AI Pipeline.

The module must:

- Conform to the pipeline's micro-tool architecture
- Follow the configuration-driven design
- Integrate with the orchestrator (`src/pipeline/orchestrator.py`)
- Remain fully local-first
- Support both CLI and Web GUI
- Maintain deterministic reproducibility
- Operate within GTX 1650 VRAM constraints

This replaces the earlier standalone FRD and aligns fully with the current repository structure.

---

# 2. Architectural Alignment

## 2.1 Pipeline Position

Album Layout remains Stage 9:

```
8. NARRATIVE ──────────── CLIP + DBSCAN scene clustering
9. LAYOUT ─────────────── Album page generation (Enhanced)
10. REPORTING ─────────── JSON report + PDF summary
```

## 2.2 Module Location

Primary Engine:

```
src/layout/engine.py
```

Supporting Components (New Submodules):

```
src/layout/
 ├── engine.py                  # Orchestrates layout generation (entry point)
 ├── template_registry.py       # JSON template loading + validation
 ├── algorithmic_generator.py   # Rule-based layout algorithms (hero, grid, collage)
 ├── background_selector.py     # Dominant-color background matching (LAB ΔE)
 ├── color_analyzer.py          # K-Means dominant color extraction per image
 ├── renderer.py                # Full-resolution JPEG spread compositor (Pillow)
 ├── project_state.py           # project.json serialization and rebuild
 ├── models.py                  # Data classes: PageLayout, CellPlacement, AlbumProject
 └── web/ (GUI layer — Phase 2)
```

The layout module must remain independent and SOLID-compliant.

---

# 3. Input Contracts (Upstream Integration)

Album Layout must consume outputs from prior stages without reprocessing.

## 3.1 Image Sources

Primary source:
```
output/<pipeline_name>/final/
```

Optional overlays (RGBA PNGs with transparency):
```
output/<pipeline_name>/cutouts/
```

Optional cropped variants:
```
output/<pipeline_name>/cropped/
```

Narrative clusters from Stage 8 must be accessible via in-memory data structure passed from orchestrator.

Layout module must NOT re-run:
- CLIP
- DBSCAN
- Color grading
- Segmentation

It only consumes results.

## 3.2 Image Discovery

The engine scans `final/` for `*.jpg`, `*.jpeg`, `*.png` files, sorted alphabetically.
For each image, it checks for a matching `cutouts/<stem>.png` if cutouts are enabled.
Aspect ratios are computed by reading image headers only (Pillow `Image.open().size`), not loading full pixel data.

---

# 4. Configuration System Integration

## 4.1 YAML Schema Extension

Extend `src/config/schema.py` with:

```yaml
layout:
  enabled: true
  mode: "template"              # template | algorithmic | mixed
  page_size: [3600, 2400]       # pixels [width, height] — 12"×8" at 300dpi
  dpi: 300
  images_per_page: 0            # 0 = auto (3–6 based on count), or fixed n
  padding: 60                   # pixels — outer margin
  gutter: 30                    # pixels — gap between cells
  use_cutouts: false            # composite RGBA cutouts over background
  background_directory: "assets/backgrounds"
  background_strategy: "dominant"   # dominant | average
  export:
    format: "jpeg"
    quality: 95
```

All values must have Pydantic validation.

## 4.2 Pydantic Models

```python
class LayoutExportConfig(BaseModel):
    format: str = "jpeg"
    quality: int = 95

class LayoutConfig(BaseModel):
    enabled: bool = False
    mode: str = "template"                    # template | algorithmic | mixed
    page_size: List[int] = Field(default_factory=lambda: [3600, 2400])
    dpi: int = 300
    images_per_page: int = 0                  # 0 = auto
    padding: int = 60
    gutter: int = 30
    use_cutouts: bool = False
    background_directory: str = "assets/backgrounds"
    background_strategy: str = "dominant"     # dominant | average
    export: LayoutExportConfig = Field(default_factory=LayoutExportConfig)
```

## 4.3 Determinism

- Sorting must remain alphabetical (from ingestion stage)
- Narrative ordering must remain deterministic
- Project JSON must capture resolved config snapshot
- No random selection of backgrounds or templates

---

# 5. Layout Engine Requirements

## 5.1 Layout Modes

Phase 1 must support:

1. **Template** — JSON-defined cell placements (primary mode)
2. **Algorithmic** — Rule-based generators:
   - `grid` — Equal-sized cells in N columns × M rows
   - `hero` — One large cell (60%+ area) + smaller supporting cells
   - `columns` — Side-by-side vertical strips
   - `triptych` — Three equal panels
3. **Mixed** — Cycles through available templates and algorithms

Mode selection is per album (from config). Per-page override is Phase 2 (GUI).

## 5.2 Template JSON Schema

Each template file defines one page layout:

```json
{
  "name": "hero_right_2",
  "description": "Hero image on the right, two stacked images on the left",
  "image_count": 3,
  "cells": [
    { "x": 0.0,   "y": 0.0,  "w": 0.40, "h": 0.50 },
    { "x": 0.0,   "y": 0.50, "w": 0.40, "h": 0.50 },
    { "x": 0.40,  "y": 0.0,  "w": 0.60, "h": 1.0  }
  ]
}
```

Cell coordinates are **normalized** (0.0–1.0). The renderer converts them to pixel coordinates using `page_size`, `padding`, and `gutter`.

## 5.3 Images-Per-Page Heuristic (auto mode)

When `images_per_page: 0`:

| Total Images | Images Per Page |
|:-------------|:----------------|
| 1–6          | 1–2             |
| 7–20         | 3–4             |
| 21–50        | 4–5             |
| 51+          | 5–6             |

This ensures albums don't become too thin or too dense.

---

# 6. Narrative-Aware Page Structuring

Layout must optionally use Stage 8 clusters.

Options:
- One cluster per page
- Multiple small clusters combined
- Manual ordering override

User-defined ordering always takes priority.

**Phase 1 implementation:** If narrative clusters exist, images within the same cluster are kept together on the same page. Clusters are processed in order.

---

# 7. Background Selection System

## 7.1 Source

Flat directory: `assets/backgrounds/` containing JPEG images.

If directory is empty or doesn't exist, use solid white (#FFFFFF) as fallback.

## 7.2 Color Extraction

For each image being placed on a page:
- Resize to 64×64 thumbnail
- Run K-Means (k=3) in LAB color space
- Extract the cluster center with the largest member count → dominant color

For each candidate background image:
- Same K-Means extraction → dominant color

## 7.3 Strategy

- **Dominant mode**: Match the page's strongest single color with the background's dominant color
- **Average mode**: Compute mean LAB color across all images on the page, match to background mean LAB

## 7.4 Scoring

For each candidate background:
1. Extract dominant color (LAB)
2. Compute **ΔE** (CIE76 Euclidean distance in LAB space) against page color
3. Select background with **minimum ΔE** score

No randomness permitted. Ties broken alphabetically by filename.

---

# 8. Cutout Compositing

If `use_cutouts: true`:

- Background image fills page canvas first
- Each cell renders the corresponding `cutouts/<stem>.png` (RGBA) over background
- Alpha compositing via `PIL.Image.alpha_composite()`
- Maintain alpha softness (already applied in Stage 5)

If false:
- Background fills canvas
- Flat graded JPEGs are placed in cells with no transparency

---

# 9. Data Models

## 9.1 CellPlacement

```python
@dataclass
class CellPlacement:
    image_path: Path              # Path to the image file
    cutout_path: Optional[Path]   # Path to RGBA cutout (if exists)
    x: int                        # Pixel X on page canvas
    y: int                        # Pixel Y on page canvas
    width: int                    # Cell width in pixels
    height: int                   # Cell height in pixels
    z_order: int = 0              # Layering order (for cutout compositing)
```

## 9.2 PageLayout

```python
@dataclass
class PageLayout:
    page_number: int
    cells: List[CellPlacement]
    background_path: Optional[Path]   # Selected background image
    background_color: Tuple[int, int, int] = (255, 255, 255)  # Fallback
    layout_mode: str = "template"
    template_name: Optional[str] = None
```

## 9.3 AlbumProject

```python
@dataclass
class AlbumProject:
    config_snapshot: dict         # Frozen config at generation time
    pages: List[PageLayout]
    total_images: int
    generation_timestamp: str
    seed: Optional[int] = None
```

---

# 10. Rendering Engine

## 10.1 Output

Only JPEG spreads.

Location:
```
output/<pipeline_name>/album/
```

Filename pattern: `page_001.jpg`, `page_002.jpg`, ...

## 10.2 Rendering Pipeline

For each `PageLayout`:

```
1. Create blank canvas (page_size, white or background_color)
2. If background_path exists → resize to fit → paste on canvas
3. For each CellPlacement (sorted by z_order):
   a. Load source image (full resolution)
   b. Resize to fit cell (width, height) maintaining aspect ratio
   c. If cutout mode → alpha_composite over canvas
   d. Else → paste at (x, y)
4. Save as JPEG with configured quality
5. Close all image handles (memory management)
```

## 10.3 Memory Management

- Process one page at a time (never hold all pages in memory)
- Load images per-page, close after rendering
- Use Pillow (CPU-bound, no VRAM required)
- Target < 2GB RAM per page at 300dpi

---

# 11. Project State Management

## 11.1 Save File

Save as:
```
output/<pipeline_name>/album/project.json
```

Must include:
- Resolved config (frozen snapshot)
- Page definitions (all CellPlacement data)
- Background selections per page
- Image paths (relative to output dir)
- Generation timestamp

## 11.2 Rebuild Capability

System must regenerate album exactly from:
- project.json
- source images

---

# 12. Reporting Integration

Stage 10 must include:

In `report.json`:
- `total_album_pages`: int
- `layout_modes_used`: list of unique modes
- `backgrounds_used`: list of background filenames
- `album_rendering_seconds`: float

In `summary.pdf`:
- Album overview section with page count

---

# 13. CLI Interface

Must integrate with existing entry point:

```
python main.py --config configs/prod_config.yaml
```

Layout stage must:
- Respect `enabled: false`
- Support `--dry-run`
- Log progress with tqdm
- Print: `"--- Stage 9: Album Layout (X pages) ---"`

---

# 14. Performance Constraints

- Handle ~1000 images
- O(n) page assignment (linear scan through sorted images)
- O(b) background scoring per page (b = number of backgrounds)
- Stream-based rendering per page (one at a time)
- CPU-only rendering via Pillow (no VRAM needed)

---

# 15. Testing Requirements

Phase 1 tests:

- `test_layout_templates.py` — Template parsing, validation, normalized cell coordinates
- `test_background_selection.py` — K-Means extraction, ΔE scoring, deterministic selection
- `test_album_rendering.py` — End-to-end render: canvas creation, image placement, JPEG output
- `test_project_state.py` — JSON serialization, deserialization, rebuild consistency

All must integrate with existing pytest structure.

---

# 16. Explicit Non-Goals

- No cloud APIs
- No PDF album export
- No template marketplace
- No client proofing
- No AI layout generator (Phase 3)
- No Web GUI (Phase 2)

---

# 17. Development Phases

**Phase 1 (Current Implementation):**
- [x] Enhanced config schema
- [ ] Template registry (JSON templates + bundled defaults)
- [ ] Algorithmic layouts (grid, hero, columns, triptych)
- [ ] Color analyzer (K-Means dominant color)
- [ ] Background selector (ΔE matching)
- [ ] Full-resolution JPEG renderer
- [ ] Project state serialization
- [ ] Orchestrator integration
- [ ] Tests

**Phase 2:**
- Web GUI editor (Fabric.js or similar)
- Project JSON persistence and reload
- Per-page mode override
- Undo/Redo

**Phase 3:**
- AI layout integration (local LLM or rule-based neural generator)
- Variant generation
- Style-aware layout suggestions

---

# Final Summary

The Album Layout module becomes a first-class micro-tool that:

- Fully respects the pipeline architecture
- Uses Narrative clustering as structural guidance
- Uses existing graded/cutout assets
- Remains config-driven
- Supports deterministic CLI and rich GUI workflows
- Preserves reproducibility
- Integrates into reporting

It upgrades Stage 9 from simple layout generation to a professional album composition engine while preserving the architectural philosophy of the pipeline.
