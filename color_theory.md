# Color Theory & AI Colorist Reference
# Wedding Photography AI Post-Production Pipeline
# ═══════════════════════════════════════════════════

This document is the **authoritative reference** for the automated color grading
system. Every adjustment value, every psychological rationale, and every
segmentation strategy is defined here. The code reads from this spec; this file
IS the colorist's brain.

---

## 1. Color Psychology & Perception

### 1.1 Emotional Associations

| Color | Emotion | Wedding Context | Handling Rule |
|:------|:--------|:----------------|:--------------|
| **Red** | Passion, love, urgency | Bouquets, lipstick, lehenga details | Protect saturation; never let it bleed into skin. Limit to accent (≤10% frame area). |
| **Blue** | Trust, calm, depth | Sky, suits, twilight | Use in shadows for cinematic separation. Shift towards teal (hue 190°) to avoid "police light" blue. |
| **Yellow** | Joy, warmth, energy | Sunlight, golden hour, décor | Critical: must not contaminate skin (causes sickly cast). Shift skin-adjacent yellows towards orange. |
| **Green** | Nature, renewal, freshness | Foliage, lawns, gardens | Almost always needs taming. Shift yellow-greens → emerald. Desaturate to let subjects pop. |
| **Orange** | Warmth, creativity, comfort | Skin tones, candlelight, autumn | The foundation of all skin tones. Protect the 15°–40° hue range at all costs. |
| **Purple** | Luxury, mystery, royalty | Twilight, decorative lighting | Enhance in shadows for mood. Dangerous in midtones (makes skin look bruised). |
| **White** | Purity, innocence | Wedding dress, tablecloths | Must remain perfectly neutral. Any color cast here is immediately visible. |
| **Black** | Elegance, power, formality | Suits, tuxedos, shadows | Deep but not crushed. Maintain shadow detail (L > 5 in LAB). |

### 1.2 Perceptual Principles

- **Equiluminant Vibration:** Two colors with identical luminance but different hue
  create visual "jitter" (the brain processes L faster than H). The pipeline must ensure
  **ΔL ≥ 15** between adjacent colored regions.
- **Warm Advance / Cool Recede:** Warm hues (H: 0°–60°, 300°–360°) visually advance.
  Cool hues (H: 150°–270°) recede. We exploit this for subject/background separation.
- **Simultaneous Contrast:** A grey patch on a red background appears greenish. The
  pipeline must account for this when adjusting neutral elements near saturated regions.

### 1.3 Palette Dynamics

| Palette Type | Saturation Range | Lightness Range | Mood | Best For |
|:-------------|:-----------------|:----------------|:-----|:---------|
| Pastel | S: 15–35% | L: 70–90% | Dreamy, soft, romantic | Bridal prep, flat-lay, details |
| Natural | S: 30–55% | L: 40–70% | Authentic, warm, honest | Documentary, reportage |
| Saturated | S: 55–80% | L: 35–65% | Bold, celebratory, vibrant | Indian weddings, receptions |
| Muted/Moody | S: 15–40% | L: 20–50% | Dramatic, editorial, cinematic | Evening, artistic portraits |
| High Key | S: 10–30% | L: 75–95% | Airy, ethereal, light | Beach weddings, minimalist |

---

## 2. Color Harmonies & Composition

### 2.1 Harmonic Models

| Harmony | Wheel Relationship | Use Case | Feeling |
|:--------|:-------------------|:---------|:--------|
| **Monochromatic** | Single hue, varied L/S | Sepia, B&W tinted, moody edits | Unified, cohesive, calm |
| **Analogous** | 3 adjacent hues (30° spread) | Golden hour (Yellow→Orange→Red) | Warm, harmonious, intimate |
| **Accented Analogous** | Analogous + 1 complement | Warm scene + blue sky accent | Balanced drama |
| **Complementary** | 2 opposites (180° apart) | Orange skin vs. Teal shadow (cinema) | High contrast, vibrant, bold |
| **Split Complementary** | Base + 2 adjacent to complement | Subject warm, bg split cool tones | Softer than complementary |
| **Triadic** | 3 equidistant (120° apart) | Colorful cultural weddings | Vibrant, diverse, playful |
| **Double Split** | 2 complementary pairs (X shape) | Complex multi-element scenes | Rich, requires careful balancing |

### 2.2 Area Weighting Rules

The visual weight of a color should follow these proportions:

```
Complementary:      70% Dominant  /  25% Complement /  5% Neutral
Analogous:          50% Primary   /  25% Support A  / 25% Support B
Split Complementary: 60% Base    /  20% Split A    / 20% Split B
Triadic:            40% Primary   /  30% Secondary  / 30% Tertiary
Accented Analogous: 40% Main     /  20% Analog A   / 20% Analog B / 20% Accent
```

---

## 3. Technical Color Models

### 3.1 Working Spaces

| Space | Channels | Use In Pipeline | Notes |
|:------|:---------|:----------------|:------|
| **RGB (sRGB)** | R, G, B | Input/Output | Web-safe. Final delivery format. Gamma-encoded. |
| **HSL** | Hue, Sat, Light | Selective adjustments | Intuitive for "shift green hue" type operations. Use OpenCV HLS (note: H is 0–180). |
| **LAB** | L, a, b | Statistical transfer, luminance isolation | Perceptually uniform. L channel = pure brightness. a = green↔red. b = blue↔yellow. |
| **HSV** | Hue, Sat, Value | Mask generation by color | Better than HSL for "select all blues" type masks. |

### 3.2 Critical Ranges (OpenCV Scale)

| Property | OpenCV HSV Range | Real-World Hue |
|:---------|:-----------------|:---------------|
| Skin Tones | H: 5–25, S: 40–170, V: 80–255 | Orange-Peach (15°–40° on 360° wheel) |
| Sky Blue | H: 95–125, S: 50–255, V: 80–255 | Azure to Cerulean |
| Vegetation | H: 30–85, S: 40–255, V: 30–200 | Yellow-green to Deep green |
| Warm Whites | H: 0–30, S: 0–40, V: 200–255 | Dress, tablecloths |

---

## 4. Image Analysis (Pre-Grading Diagnostics)

Before applying any grade, the pipeline MUST analyze:

### 4.1 Luminance Analysis
- **Histogram Shape:** Check for clipping at 0 (crushed blacks) or 255 (blown highlights).
- **Dynamic Range Score:** `DR = percentile_99 - percentile_1`. If DR < 150, image is flat → needs contrast.
- **Key Detection:** Mean luminance < 85 = Low Key. Mean > 170 = High Key. Between = Normal.

### 4.2 Color Analysis
- **Dominant Hue:** Weighted histogram peak in HSV H-channel.
- **Skin Presence:** Detect pixels in skin-tone HSV range. If > 5% of frame, enable skin protection.
- **Color Temperature:** Estimate via grey-world: `temp_bias = mean(R) - mean(B)`. Positive = warm. Negative = cool.
- **Saturation Distribution:** `mean(S)` across frame. Low (<30) = desaturated/flat. High (>60) = vivid.

### 4.3 Vectorscope Emulation
Compute 2D histogram of `a` vs `b` channels in LAB space. Check if the mass falls along
the skin-tone line (roughly 45° in a-b space). If it deviates, the white balance is off.

---

## 5. Filter Presets (Style Profiles)

Each preset defines specific numerical adjustments applied during grading.
The config key `color_grading.style` selects the active preset.

### 5.1 Preset: `natural`
*Goal: Accurate, clean, true-to-life with slight warmth.*

```yaml
global:
  temperature_shift: +3          # Kelvin shift (slight warm)
  tint_shift: 0
  exposure_offset: 0.0
  contrast: 1.05                 # Gentle S-curve
  saturation_scale: 1.0          # No change
  vibrance_scale: 1.05           # Slight boost to muted colors
  
tone_curve:
  shadows_lift: 0                # Pure black
  highlights_roll: 0             # Pure white
  midtone_gamma: 1.0

split_tone:
  shadow_hue: null               # No split toning
  highlight_hue: null

per_channel_hsl:                 # Offsets from base
  red:    { hue: 0, sat: 0,   lum: 0 }
  orange: { hue: 0, sat: 0,   lum: 0 }
  yellow: { hue: 0, sat: -5,  lum: 0 }
  green:  { hue: 0, sat: -10, lum: -5 }
  blue:   { hue: 0, sat: 0,   lum: 0 }
  purple: { hue: 0, sat: 0,   lum: 0 }

vignette: { strength: 0, radius: 0.8 }
grain:    { amount: 0, size: 1.0 }
```

### 5.2 Preset: `cinematic`
*Goal: Teal-and-orange complementary push. Lifted blacks. Rolled highlights. Film-like.*

```yaml
global:
  temperature_shift: +5
  tint_shift: -2
  exposure_offset: -0.1
  contrast: 1.15
  saturation_scale: 0.90         # Slightly desaturated globally
  vibrance_scale: 1.10

tone_curve:
  shadows_lift: 15               # Lifted/faded blacks (0→15)
  highlights_roll: -10           # Rolled-off highlights (255→245)
  midtone_gamma: 0.95            # Slight darken midtones

split_tone:
  shadow_hue: 200                # Teal shadows (H=200°)
  shadow_saturation: 25
  highlight_hue: 35              # Warm orange highlights (H=35°)
  highlight_saturation: 20

per_channel_hsl:
  red:    { hue: +3,  sat: -5,  lum: 0 }
  orange: { hue: -2,  sat: +5,  lum: +5 }    # Protect/boost skin warmth
  yellow: { hue: -10, sat: -15, lum: -5 }     # Push yellows towards orange
  green:  { hue: +15, sat: -25, lum: -10 }    # Shift green → teal, desaturate
  blue:   { hue: -10, sat: +10, lum: -10 }    # Shift blue → teal, boost
  purple: { hue: +5,  sat: -10, lum: -5 }

vignette: { strength: 0.3, radius: 0.7 }
grain:    { amount: 8, size: 1.5 }
```

### 5.3 Preset: `pastel`
*Goal: Soft, dreamy, low-contrast. Lifted shadows, desaturated, high lightness.*

```yaml
global:
  temperature_shift: +2
  tint_shift: +2
  exposure_offset: +0.3
  contrast: 0.85                 # Reduced contrast
  saturation_scale: 0.65         # Significant desaturation
  vibrance_scale: 0.80

tone_curve:
  shadows_lift: 30               # Very lifted blacks (dreamy fade)
  highlights_roll: -5
  midtone_gamma: 1.10            # Brighter midtones

split_tone:
  shadow_hue: 220                # Soft lavender shadows
  shadow_saturation: 15
  highlight_hue: 45              # Peachy highlights
  highlight_saturation: 12

per_channel_hsl:
  red:    { hue: +5,  sat: -20, lum: +10 }
  orange: { hue: 0,   sat: -15, lum: +10 }
  yellow: { hue: -5,  sat: -20, lum: +15 }
  green:  { hue: +10, sat: -30, lum: +10 }
  blue:   { hue: +5,  sat: -20, lum: +10 }
  purple: { hue: 0,   sat: -15, lum: +10 }

vignette: { strength: 0, radius: 0.8 }
grain:    { amount: 3, size: 1.0 }
```

### 5.4 Preset: `moody`
*Goal: Dark, dramatic, editorial. Deep shadows, muted palette, strong contrast.*

```yaml
global:
  temperature_shift: -3
  tint_shift: -3
  exposure_offset: -0.4
  contrast: 1.25
  saturation_scale: 0.70
  vibrance_scale: 0.85

tone_curve:
  shadows_lift: 5
  highlights_roll: -20           # Significant highlight rolloff
  midtone_gamma: 0.85            # Darker midtones

split_tone:
  shadow_hue: 230                # Deep blue shadows
  shadow_saturation: 20
  highlight_hue: 40
  highlight_saturation: 10

per_channel_hsl:
  red:    { hue: 0,   sat: -10, lum: -10 }
  orange: { hue: -3,  sat: -5,  lum: -5 }
  yellow: { hue: -8,  sat: -20, lum: -15 }
  green:  { hue: +10, sat: -30, lum: -20 }
  blue:   { hue: -5,  sat: +5,  lum: -15 }
  purple: { hue: +5,  sat: +10, lum: -10 }

vignette: { strength: 0.5, radius: 0.6 }
grain:    { amount: 12, size: 2.0 }
```

### 5.5 Preset: `golden_hour`
*Goal: Warm, amber tones. Simulates late afternoon light.*

```yaml
global:
  temperature_shift: +12
  tint_shift: +3
  exposure_offset: +0.1
  contrast: 1.10
  saturation_scale: 1.10
  vibrance_scale: 1.15

tone_curve:
  shadows_lift: 8
  highlights_roll: -5
  midtone_gamma: 1.05

split_tone:
  shadow_hue: 30                 # Warm amber shadows
  shadow_saturation: 20
  highlight_hue: 45              # Golden highlights
  highlight_saturation: 25

per_channel_hsl:
  red:    { hue: -3,  sat: +10, lum: +5 }
  orange: { hue: -5,  sat: +15, lum: +10 }
  yellow: { hue: -8,  sat: +10, lum: +10 }
  green:  { hue: -15, sat: -20, lum: -5 }     # Push greens warm
  blue:   { hue: 0,   sat: -15, lum: -10 }
  purple: { hue: -10, sat: -10, lum: -5 }

vignette: { strength: 0.2, radius: 0.75 }
grain:    { amount: 5, size: 1.2 }
```

### 5.6 Preset: `film_kodak`
*Goal: Emulate Kodak Portra 400. Warm skin, muted greens, subtle grain.*

```yaml
global:
  temperature_shift: +4
  tint_shift: +1
  exposure_offset: 0.0
  contrast: 1.08
  saturation_scale: 0.92
  vibrance_scale: 1.05

tone_curve:
  shadows_lift: 12               # Slight fade
  highlights_roll: -8
  midtone_gamma: 1.02

split_tone:
  shadow_hue: 160                # Subtle green-teal shadow
  shadow_saturation: 8
  highlight_hue: 40              # Warm highlight
  highlight_saturation: 15

per_channel_hsl:
  red:    { hue: +2,  sat: -5,  lum: +3 }
  orange: { hue: -3,  sat: +8,  lum: +5 }     # Beautiful skin rendition
  yellow: { hue: -5,  sat: -10, lum: +3 }
  green:  { hue: +8,  sat: -20, lum: -8 }     # Muted greens (Portra signature)
  blue:   { hue: -8,  sat: -5,  lum: 0 }
  purple: { hue: +3,  sat: -8,  lum: -3 }

vignette: { strength: 0.15, radius: 0.8 }
grain:    { amount: 10, size: 1.8 }            # Film grain
```

### 5.7 Preset: `film_fuji`
*Goal: Emulate Fujifilm Pro 400H. Cool shadows, clean highlights, lifted blacks.*

```yaml
global:
  temperature_shift: -2
  tint_shift: +2
  exposure_offset: +0.15
  contrast: 1.05
  saturation_scale: 0.88
  vibrance_scale: 1.08

tone_curve:
  shadows_lift: 18
  highlights_roll: -5
  midtone_gamma: 1.05

split_tone:
  shadow_hue: 185                # Cyan-teal shadows (Fuji signature)
  shadow_saturation: 15
  highlight_hue: 50
  highlight_saturation: 10

per_channel_hsl:
  red:    { hue: +2,  sat: -8,  lum: +2 }
  orange: { hue: 0,   sat: -3,  lum: +5 }
  yellow: { hue: -3,  sat: -12, lum: +5 }
  green:  { hue: +5,  sat: -15, lum: -3 }
  blue:   { hue: -5,  sat: +8,  lum: +3 }     # Enhanced cool tones
  purple: { hue: -5,  sat: -5,  lum: 0 }

vignette: { strength: 0.1, radius: 0.85 }
grain:    { amount: 6, size: 1.4 }
```

### 5.8 Preset: `vibrant`
*Goal: Punchy, Instagram-ready. High saturation, strong contrast, clean colors.*

```yaml
global:
  temperature_shift: +3
  tint_shift: 0
  exposure_offset: +0.05
  contrast: 1.20
  saturation_scale: 1.30
  vibrance_scale: 1.25

tone_curve:
  shadows_lift: 0
  highlights_roll: 0
  midtone_gamma: 0.98

split_tone:
  shadow_hue: null
  highlight_hue: null

per_channel_hsl:
  red:    { hue: 0,   sat: +15, lum: 0 }
  orange: { hue: -3,  sat: +10, lum: +5 }
  yellow: { hue: 0,   sat: +10, lum: +5 }
  green:  { hue: -5,  sat: +10, lum: -5 }
  blue:   { hue: 0,   sat: +15, lum: -5 }
  purple: { hue: 0,   sat: +10, lum: 0 }

vignette: { strength: 0.1, radius: 0.8 }
grain:    { amount: 0, size: 1.0 }
```

### 5.9 Preset: `black_and_white`
*Goal: Rich monochrome with tonal control per original color.*

```yaml
global:
  temperature_shift: 0
  tint_shift: 0
  exposure_offset: 0.0
  contrast: 1.20
  saturation_scale: 0.0          # Complete desaturation
  vibrance_scale: 0.0

tone_curve:
  shadows_lift: 5
  highlights_roll: -5
  midtone_gamma: 0.95

split_tone:
  shadow_hue: null
  highlight_hue: null

# These control luminance contribution of each original color
per_channel_hsl:
  red:    { hue: 0, sat: 0, lum: +10 }        # Brighten skin slightly
  orange: { hue: 0, sat: 0, lum: +15 }        # Glow on skin
  yellow: { hue: 0, sat: 0, lum: +5 }
  green:  { hue: 0, sat: 0, lum: -15 }        # Darken foliage for drama
  blue:   { hue: 0, sat: 0, lum: -20 }        # Darken skies
  purple: { hue: 0, sat: 0, lum: -10 }

vignette: { strength: 0.3, radius: 0.7 }
grain:    { amount: 15, size: 2.5 }           # Heavy film grain
```

---

## 6. Semantic Segmentation Strategy (SAM Integration)

### 6.1 Segmentation Classes

The pipeline uses SAM (Segment Anything Model) or a lightweight semantic segmenter
(e.g., SegFormer) to produce masks for each region. Adjustments in Section 5 are
applied globally first, then the following **per-region overrides** are layered on top.

| Class ID | Label | SAM Prompts | Priority |
|:---------|:------|:------------|:---------|
| 0 | `skin` | "person", "face", "hand", "arm" | **HIGHEST** - Never degrade skin |
| 1 | `sky` | "sky", "clouds" | HIGH - Major mood contributor |
| 2 | `vegetation` | "tree", "grass", "bush", "plant", "leaf" | MEDIUM - Usually needs taming |
| 3 | `ground` | "floor", "road", "sand", "path" | MEDIUM |
| 4 | `architecture` | "building", "wall", "pillar", "arch" | MEDIUM |
| 5 | `dress_white` | "white dress", "veil", "gown" | HIGH - Must stay neutral |
| 6 | `suit_dark` | "suit", "tuxedo", "dark clothing" | MEDIUM |
| 7 | `decor` | "flowers", "candles", "lights", "table" | LOW - Enhance or leave |
| 8 | `background` | Everything not classified above | LOW |

### 6.2 Per-Region Adjustment Overrides

These are **deltas** applied ON TOP of the selected filter preset adjustments.
They ensure that regardless of the chosen style, skin stays healthy and key
elements remain visually correct.

#### Skin Protection Layer (Class 0)
```
- hue_clamp: [15, 40]           # Force skin into healthy orange range (360° scale)
- saturation_limit: 65          # Cap saturation to prevent oversaturated skin
- luminance_boost: +5%          # Slight glow
- red_channel_clean: true       # Remove magenta from skin by reducing blue in red channel
- temperature_local: +3K        # Always slightly warm, even in cool presets
```

#### Sky Enhancement (Class 1)
```
- luminance_offset: -10%        # Recover highlights
- saturation_boost: +10%        # Add depth
- hue_shift: towards 195°       # Push generic blue → teal-azure
- gradient_aware: true           # Apply more at top, less at horizon
```

#### Vegetation Taming (Class 2)
```
- hue_shift: +15°               # Yellow-green → Emerald
- saturation_offset: -20%       # Reduce competition with subject
- luminance_offset: -8%         # Slightly darker
- shadow_spread: true           # Spread hue in shadow regions for richness
```

#### White Dress Protection (Class 5)
```
- saturation_clamp: 10          # Nearly zero saturation
- luminance_protect: true       # Never clip highlights; recover if blown
- cast_removal: true            # Neutralize any color cast (target a=128, b=128 in LAB)
```

#### Dark Suit Enhancement (Class 6)
```
- black_point: L=8              # Deep but not crushed (LAB L channel)
- cast_removal: true            # Remove blue/brown tints
- local_contrast: +10%          # Reveal fabric texture in shadows
```

### 6.3 Workflow: Segment → Analyze → Grade → Blend

```
1. SEGMENT:  Run segmenter → produce N binary masks
2. ANALYZE:  For each mask, compute HSL statistics (mean H, S, L)
3. GRADE:    Apply global preset (Section 5)
4. OVERRIDE: For each mask, apply per-region delta (Section 6.2)
5. BLEND:    Feather mask edges (Gaussian blur σ=5px) to avoid harsh transitions
6. MERGE:    Composite all adjusted regions back into the frame
7. FINALIZE: Apply vignette + grain from preset
```

---

## 7. Processing Order & Pipeline Integration

The color engine must execute in this exact order:

```
Input Image (RGB, 8/16-bit)
  │
  ├─ 1. WHITE BALANCE CORRECTION (Global)
  │     └─ Grey-world or reference-based
  │
  ├─ 2. EXPOSURE NORMALIZATION (Global)
  │     └─ Histogram stretch with soft clip
  │
  ├─ 3. SEGMENTATION (SAM / SegFormer)
  │     └─ Generate masks for skin, sky, vegetation, dress, suit, background
  │
  ├─ 4. TONE CURVE APPLICATION (Global)
  │     └─ shadows_lift, highlights_roll, midtone_gamma
  │
  ├─ 5. CONTRAST ADJUSTMENT (Global)
  │     └─ S-curve in LAB L-channel
  │
  ├─ 6. PER-CHANNEL HSL ADJUSTMENTS (Global then Local)
  │     ├─ Apply preset per_channel_hsl globally
  │     └─ Apply per-region overrides using masks
  │
  ├─ 7. SPLIT TONING (Global)
  │     └─ Colorize shadows and highlights independently
  │
  ├─ 8. SATURATION / VIBRANCE (Global)
  │     └─ saturation_scale on all, vibrance on under-saturated pixels
  │
  ├─ 9. SKIN PROTECTION PASS (Local, Mask-based)
  │     └─ Clamp skin hue/sat, boost luminance, clean channels
  │
  ├─ 10. VIGNETTE (Global)
  │     └─ Radial darkening from center
  │
  └─ 11. GRAIN (Global)
        └─ Gaussian noise overlay
```

---

## 8. Quality Assurance Checks

After grading, the pipeline validates:

| Check | Condition | Action if Failed |
|:------|:----------|:-----------------|
| Skin hue in range | H ∈ [15°, 40°] for skin pixels | Shift back towards 25° |
| Highlight clipping | < 0.5% pixels at 255 | Reduce exposure |
| Shadow clipping | < 0.5% pixels at 0 | Lift shadows |
| White dress neutral | mean(S) < 15 for dress mask | Desaturate further |
| Overall saturation | mean(S) ∈ [20, 75] | Scale towards center |
| Color temperature | Skin matches reference range | Re-correct WB |
