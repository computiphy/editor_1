"""
3D LUT Engine with Tetrahedral Interpolation
==============================================
Loads .cube LUT files and applies them to images using tetrahedral
interpolation for mathematically perfect color gradients.

Why tetrahedral over trilinear?
  - Trilinear interpolation divides each cube into 8 sub-cubes and
    interpolates within each. This causes visible banding at the
    boundaries between sub-cubes, especially in smooth gradients
    of saturated colors.
  - Tetrahedral interpolation divides each cube into 6 tetrahedra,
    providing smoother transitions and eliminating banding artifacts.

.cube format reference:
  https://resolve.training/cube-lut-specification/
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Union


def parse_cube_file(filepath: Union[str, Path]) -> Tuple[np.ndarray, int]:
    """
    Parse a .cube LUT file.

    The .cube format stores a 3D LUT as a flat list of RGB triplets,
    ordered with Red varying fastest, then Green, then Blue.

    Args:
        filepath: Path to the .cube file.

    Returns:
        (lut, size) where:
          - lut: float32 array of shape (size, size, size, 3)
          - size: Grid size (e.g., 33 for a 33x33x33 LUT)
    """
    filepath = Path(filepath)
    size = None
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse metadata
            if line.startswith('TITLE'):
                continue
            if line.startswith('DOMAIN_MIN'):
                continue
            if line.startswith('DOMAIN_MAX'):
                continue
            if line.startswith('LUT_3D_SIZE'):
                size = int(line.split()[-1])
                continue
            if line.startswith('LUT_1D_SIZE'):
                raise ValueError("1D LUTs are not supported. Use a 3D .cube file.")

            # Parse data line (three floats)
            parts = line.split()
            if len(parts) == 3:
                try:
                    r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                    data.append([r, g, b])
                except ValueError:
                    continue

    if size is None:
        raise ValueError(f"No LUT_3D_SIZE found in {filepath}")

    expected = size ** 3
    if len(data) != expected:
        raise ValueError(
            f"Expected {expected} data points for size {size}, got {len(data)}"
        )

    # Reshape: .cube format is R-fastest, then G, then B
    # data[r + g*size + b*size*size] = (R_out, G_out, B_out)
    # NumPy C-order reshape gives lut[b,g,r,3], so transpose to lut[r,g,b,3]
    lut = np.array(data, dtype=np.float32).reshape(size, size, size, 3)
    lut = lut.transpose(2, 1, 0, 3).copy()  # (B,G,R,3) → (R,G,B,3)

    return lut, size


def apply_lut3d(image: np.ndarray, lut_path: Union[str, Path],
                intensity: float = 1.0) -> np.ndarray:
    """
    Apply a 3D LUT to an image using tetrahedral interpolation.

    Args:
        image: float32 array (H, W, 3) in [0-1] range.
        lut_path: Path to a .cube LUT file.
        intensity: Blend factor (0 = original, 1 = full LUT effect).

    Returns:
        float32 array (H, W, 3) in [0-1] range.
    """
    lut, size = parse_cube_file(lut_path)
    return apply_lut3d_array(image, lut, size, intensity)


def apply_lut3d_array(image: np.ndarray, lut: np.ndarray, size: int,
                      intensity: float = 1.0) -> np.ndarray:
    """
    Apply a pre-loaded 3D LUT array to an image using tetrahedral interpolation.

    Args:
        image: float32 array (H, W, 3) in [0-1] range.
        lut: float32 array (size, size, size, 3).
        size: Grid dimension.
        intensity: Blend factor (0 = original, 1 = full LUT effect).

    Returns:
        float32 array (H, W, 3) in [0-1] range.
    """
    h, w = image.shape[:2]
    img_flat = np.clip(image.reshape(-1, 3), 0.0, 1.0).astype(np.float64)

    # Scale to LUT indices
    scale = size - 1
    rgb_scaled = img_flat * scale

    # Integer indices (floor)
    r0 = np.floor(rgb_scaled[:, 0]).astype(np.int32)
    g0 = np.floor(rgb_scaled[:, 1]).astype(np.int32)
    b0 = np.floor(rgb_scaled[:, 2]).astype(np.int32)

    # Clamp to valid range
    r0 = np.clip(r0, 0, size - 2)
    g0 = np.clip(g0, 0, size - 2)
    b0 = np.clip(b0, 0, size - 2)

    # Next indices
    r1 = r0 + 1
    g1 = g0 + 1
    b1 = b0 + 1

    # Fractional parts
    dr = rgb_scaled[:, 0] - r0
    dg = rgb_scaled[:, 1] - g0
    db = rgb_scaled[:, 2] - b0

    # ── Tetrahedral Interpolation ──────────────────────────────
    # Each cube cell is divided into 6 tetrahedra based on the
    # ordering of the fractional components (dr, dg, db).
    # This provides smoother interpolation than trilinear.

    n = len(img_flat)
    result = np.zeros((n, 3), dtype=np.float64)

    # Fetch all 8 corners of the cube for each pixel
    c000 = lut[r0, g0, b0].astype(np.float64)
    c100 = lut[r1, g0, b0].astype(np.float64)
    c010 = lut[r0, g1, b0].astype(np.float64)
    c110 = lut[r1, g1, b0].astype(np.float64)
    c001 = lut[r0, g0, b1].astype(np.float64)
    c101 = lut[r1, g0, b1].astype(np.float64)
    c011 = lut[r0, g1, b1].astype(np.float64)
    c111 = lut[r1, g1, b1].astype(np.float64)

    # Determine which tetrahedron each pixel falls into
    # 6 cases based on ordering of dr, dg, db
    dr3 = dr[:, np.newaxis]
    dg3 = dg[:, np.newaxis]
    db3 = db[:, np.newaxis]

    # Case 1: dr >= dg >= db
    mask = (dr >= dg) & (dg >= db)
    if mask.any():
        result[mask] = (c000[mask]
                        + dr3[mask] * (c100[mask] - c000[mask])
                        + dg3[mask] * (c110[mask] - c100[mask])
                        + db3[mask] * (c111[mask] - c110[mask]))

    # Case 2: dr >= db >= dg
    mask = (dr >= db) & (db >= dg)
    if mask.any():
        result[mask] = (c000[mask]
                        + dr3[mask] * (c100[mask] - c000[mask])
                        + db3[mask] * (c101[mask] - c100[mask])
                        + dg3[mask] * (c111[mask] - c101[mask]))

    # Case 3: db >= dr >= dg
    mask = (db >= dr) & (dr >= dg)
    if mask.any():
        result[mask] = (c000[mask]
                        + db3[mask] * (c001[mask] - c000[mask])
                        + dr3[mask] * (c101[mask] - c001[mask])
                        + dg3[mask] * (c111[mask] - c101[mask]))

    # Case 4: dg >= dr >= db
    mask = (dg >= dr) & (dr >= db)
    if mask.any():
        result[mask] = (c000[mask]
                        + dg3[mask] * (c010[mask] - c000[mask])
                        + dr3[mask] * (c110[mask] - c010[mask])
                        + db3[mask] * (c111[mask] - c110[mask]))

    # Case 5: dg >= db >= dr
    mask = (dg >= db) & (db >= dr)
    if mask.any():
        result[mask] = (c000[mask]
                        + dg3[mask] * (c010[mask] - c000[mask])
                        + db3[mask] * (c011[mask] - c010[mask])
                        + dr3[mask] * (c111[mask] - c011[mask]))

    # Case 6: db >= dg >= dr
    mask = (db >= dg) & (dg >= dr)
    if mask.any():
        result[mask] = (c000[mask]
                        + db3[mask] * (c001[mask] - c000[mask])
                        + dg3[mask] * (c011[mask] - c001[mask])
                        + dr3[mask] * (c111[mask] - c011[mask]))

    result = result.reshape(h, w, 3).astype(np.float32)

    # Blend with original based on intensity
    if abs(intensity - 1.0) > 0.01:
        result = image * (1.0 - intensity) + result * intensity

    return np.clip(result, 0.0, 1.0).astype(np.float32)
