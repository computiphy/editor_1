# Color Theory & AI Colorist Reference
# Wedding Photography AI Post-Production Pipeline
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

This document is the **authoritative reference** for the automated color grading
system. Every adjustment value, every psychological rationale, and every
segmentation strategy is defined here. The code reads from this spec; this file
IS the colorist's brain.

---

## 1. Color Psychology & Perception

### 1.1 Emotional Associations

| Color | Emotion | Wedding Context | Handling Rule |
|:------|:--------|:----------------|:--------------|
| **Red** | Passion, love, urgency | Bouquets, lipstick, lehenga details | Protect saturation; never let it bleed into skin. Limit to accent (â‰¤10% frame area). |
| **Blue** | Trust, calm, depth | Sky, suits, twilight | Use in shadows for cinematic separation. Shift towards teal (hue 190Â°) to avoid "police light" blue. |
| **Yellow** | Joy, warmth, energy | Sunlight, golden hour, dÃ©cor | Critical: must not contaminate skin (causes sickly cast). Shift skin-adjacent yellows towards orange. |
| **Green** | Nature, renewal, freshness | Foliage, lawns, gardens | Almost always needs taming. Shift yellow-greens â†’ emerald. Desaturate to let subjects pop. |
| **Orange** | Warmth, creativity, comfort | Skin tones, candlelight, autumn | The foundation of all skin tones. Protect the 15Â°â€“40Â° hue range at all costs. |
| **Purple** | Luxury, mystery, royalty | Twilight, decorative lighting | Enhance in shadows for mood. Dangerous in midtones (makes skin look bruised). |
| **White** | Purity, innocence | Wedding dress, tablecloths | Must remain perfectly neutral. Any color cast here is immediately visible. |
| **Black** | Elegance, power, formality | Suits, tuxedos, shadows | Deep but not crushed. Maintain shadow detail (L > 5 in LAB). |

### 1.2 Perceptual Principles

- **Equiluminant Vibration:** Two colors with identical luminance but different hue
  create visual "jitter" (the brain processes L faster than H). The pipeline must ensure
  **Î”L â‰¥ 15** between adjacent colored regions.
- **Warm Advance / Cool Recede:** Warm hues (H: 0Â°â€“60Â°, 300Â°â€“360Â°) visually advance.
  Cool hues (H: 150Â°â€“270Â°) recede. We exploit this for subject/background separation.
- **Simultaneous Contrast:** A grey patch on a red background appears greenish. The
  pipeline must account for this when adjusting neutral elements near saturated regions.

### 1.3 Palette Dynamics

| Palette Type | Saturation Range | Lightness Range | Mood | Best For |
|:-------------|:-----------------|:----------------|:-----|:---------|
| Pastel | S: 15â€“35% | L: 70â€“90% | Dreamy, soft, romantic | Bridal prep, flat-lay, details |
| Natural | S: 30â€“55% | L: 40â€“70% | Authentic, warm, honest | Documentary, reportage |
| Saturated | S: 55â€“80% | L: 35â€“65% | Bold, celebratory, vibrant | Indian weddings, receptions |
| Muted/Moody | S: 15â€“40% | L: 20â€“50% | Dramatic, editorial, cinematic | Evening, artistic portraits |
| High Key | S: 10â€“30% | L: 75â€“95% | Airy, ethereal, light | Beach weddings, minimalist |

---

## 2. Color Harmonies & Composition

### 2.1 Harmonic Models

| Harmony | Wheel Relationship | Use Case | Feeling |
|:--------|:-------------------|:---------|:--------|
| **Monochromatic** | Single hue, varied L/S | Sepia, B&W tinted, moody edits | Unified, cohesive, calm |
| **Analogous** | 3 adjacent hues (30Â° spread) | Golden hour (Yellowâ†’Orangeâ†’Red) | Warm, harmonious, intimate |
| **Accented Analogous** | Analogous + 1 complement | Warm scene + blue sky accent | Balanced drama |
| **Complementary** | 2 opposites (180Â° apart) | Orange skin vs. Teal shadow (cinema) | High contrast, vibrant, bold |
| **Split Complementary** | Base + 2 adjacent to complement | Subject warm, bg split cool tones | Softer than complementary |
| **Triadic** | 3 equidistant (120Â° apart) | Colorful cultural weddings | Vibrant, diverse, playful |
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
| **HSL** | Hue, Sat, Light | Selective adjustments | Intuitive for "shift green hue" type operations. Use OpenCV HLS (note: H is 0â€“180). |
| **LAB** | L, a, b | Statistical transfer, luminance isolation | Perceptually uniform. L channel = pure brightness. a = greenâ†”red. b = blueâ†”yellow. |
| **HSV** | Hue, Sat, Value | Mask generation by color | Better than HSL for "select all blues" type masks. |

### 3.2 Critical Ranges (OpenCV Scale)

| Property | OpenCV HSV Range | Real-World Hue |
|:---------|:-----------------|:---------------|
| Skin Tones | H: 5â€“25, S: 40â€“170, V: 80â€“255 | Orange-Peach (15Â°â€“40Â° on 360Â° wheel) |
| Sky Blue | H: 95â€“125, S: 50â€“255, V: 80â€“255 | Azure to Cerulean |
| Vegetation | H: 30â€“85, S: 40â€“255, V: 30â€“200 | Yellow-green to Deep green |
| Warm Whites | H: 0â€“30, S: 0â€“40, V: 200â€“255 | Dress, tablecloths |

---

## 4. Image Analysis (Pre-Grading Diagnostics)

Before applying any grade, the pipeline MUST analyze:

### 4.1 Luminance Analysis
- **Histogram Shape:** Check for clipping at 0 (crushed blacks) or 255 (blown highlights).
- **Dynamic Range Score:** `DR = percentile_99 - percentile_1`. If DR < 150, image is flat â†’ needs contrast.
- **Key Detection:** Mean luminance < 85 = Low Key. Mean > 170 = High Key. Between = Normal.

### 4.2 Color Analysis
- **Dominant Hue:** Weighted histogram peak in HSV H-channel.
- **Skin Presence:** Detect pixels in skin-tone HSV range. If > 5% of frame, enable skin protection.
- **Color Temperature:** Estimate via grey-world: `temp_bias = mean(R) - mean(B)`. Positive = warm. Negative = cool.
- **Saturation Distribution:** `mean(S)` across frame. Low (<30) = desaturated/flat. High (>60) = vivid.

### 4.3 Vectorscope Emulation
Compute 2D histogram of `a` vs `b` channels in LAB space. Check if the mass falls along
the skin-tone line (roughly 45Â° in a-b space). If it deviates, the white balance is off.

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
  shadows_lift: 15               # Lifted/faded blacks (0â†’15)
  highlights_roll: -10           # Rolled-off highlights (255â†’245)
  midtone_gamma: 0.95            # Slight darken midtones

split_tone:
  shadow_hue: 200                # Teal shadows (H=200Â°)
  shadow_saturation: 25
  highlight_hue: 35              # Warm orange highlights (H=35Â°)
  highlight_saturation: 20

per_channel_hsl:
  red:    { hue: +3,  sat: -5,  lum: 0 }
  orange: { hue: -2,  sat: +5,  lum: +5 }    # Protect/boost skin warmth
  yellow: { hue: -10, sat: -15, lum: -5 }     # Push yellows towards orange
  green:  { hue: +15, sat: -25, lum: -10 }    # Shift green â†’ teal, desaturate
  blue:   { hue: -10, sat: +10, lum: -10 }    # Shift blue â†’ teal, boost
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

### 5.10 Preset: `moody_forest`
*Goal: Deep, rich greens and cool tones. Analogous harmony focusing on Green/Blue.*

```yaml
global:
  temperature_shift: -5
  tint_shift: +2
  exposure_offset: -0.2
  contrast: 1.10
  saturation_scale: 0.85
  vibrance_scale: 1.0

tone_curve:
  shadows_lift: 5
  highlights_roll: -15
  midtone_gamma: 0.90

split_tone:
  shadow_hue: 210                # Deep Blue/Cyan
  shadow_saturation: 10
  highlight_hue: null

per_channel_hsl:
  red:    { hue: 0,   sat: 0,   lum: 0 }
  orange: { hue: 0,   sat: +5,  lum: +5 }    # Keep skin visible
  yellow: { hue: -10, sat: -20, lum: -5 }
  green:  { hue: +10, sat: -40, lum: -15 }   # Dark, desaturated, cool greens
  blue:   { hue: -5,  sat: -10, lum: -10 }
  purple: { hue: 0,   sat: -20, lum: 0 }

vignette: { strength: 0.4, radius: 0.6 }
grain:    { amount: 5, size: 1.2 }
```

### 5.11 Preset: `golden_hour_portrait`
*Goal: Warm, inviting, soft. Analogous harmony (Red, Orange, Yellow).*

```yaml
global:
  temperature_shift: +8
  tint_shift: +1
  exposure_offset: +0.1
  contrast: 1.0
  saturation_scale: 1.1
  vibrance_scale: 1.1

tone_curve:
  shadows_lift: 10
  highlights_roll: -5
  midtone_gamma: 1.05

split_tone:
  shadow_hue: 40                 # Warm Chocolate shadows
  shadow_saturation: 15
  highlight_hue: 50              # Golden highlights
  highlight_saturation: 10

per_channel_hsl:
  red:    { hue: +5,  sat: +5,  lum: 0 }
  orange: { hue: 0,   sat: +10, lum: +5 }
  yellow: { hue: -5,  sat: +15, lum: 0 }     # Golden sun look
  green:  { hue: -10, sat: +5,  lum: 0 }     # Warmer greens
  blue:   { hue: -10, sat: -15, lum: +5 }    # Cyan sky
  purple: { hue: 0,   sat: 0,   lum: 0 }

vignette: { strength: 0.2, radius: 0.9 }
grain:    { amount: 0, size: 1.0 }
```

### 5.12 Preset: `urban_cyberpunk`
*Goal: Neon look. Split Complementary (Purple/Pink vs. Cyan).*

```yaml
global:
  temperature_shift: -10
  tint_shift: +15                # Strong Magenta push
  exposure_offset: 0.0
  contrast: 1.20
  saturation_scale: 1.2
  vibrance_scale: 1.3

tone_curve:
  shadows_lift: 5
  highlights_roll: 0
  midtone_gamma: 0.95

split_tone:
  shadow_hue: 280                # Purple shadows
  shadow_saturation: 30
  highlight_hue: 180             # Cyan highlights
  highlight_saturation: 20

per_channel_hsl:
  red:    { hue: +10, sat: +10, lum: 0 }     # Push toward pink
  orange: { hue: 0,   sat: 0,   lum: +10 }
  yellow: { hue: -20, sat: -10, lum: 0 }
  green:  { hue: +40, sat: -20, lum: 0 }     # Push green to cyan
  blue:   { hue: -10, sat: +20, lum: +10 }   # Electric blue
  purple: { hue: +10, sat: +25, lum: +5 }    # Neon purple

vignette: { strength: 0.4, radius: 0.6 }
grain:    { amount: 0, size: 1.0 }
```

### 5.13 Preset: `vintage_painterly`
*Goal: Warm, mid-centric, low dynamic range. Mimics old masters.*

```yaml
global:
  temperature_shift: +4
  tint_shift: +5
  exposure_offset: -0.1
  contrast: 0.95
  saturation_scale: 0.85
  vibrance_scale: 0.9

tone_curve:
  shadows_lift: 25               # Faded blacks
  highlights_roll: -20           # Yellowed whites
  midtone_gamma: 1.0

split_tone:
  shadow_hue: 210                # Faint Prussian Blue
  shadow_saturation: 8
  highlight_hue: 45              # Antique Gold
  highlight_saturation: 15

per_channel_hsl:
  red:    { hue: 0,   sat: -10, lum: -5 }
  orange: { hue: -2,  sat: -5,  lum: 0 }
  yellow: { hue: -5,  sat: -10, lum: -5 }    # Ocher tones
  green:  { hue: -10, sat: -30, lum: -10 }   # Olive greens
  blue:   { hue: -5,  sat: -20, lum: -10 }
  purple: { hue: 0,   sat: -30, lum: 0 }

vignette: { strength: 0.2, radius: 0.5 }
grain:    { amount: 15, size: 2.0 }
```

### 5.14 Preset: `high_fashion`
*Goal: Bold, high contrast, clean. Double Split Complementary.*

```yaml
global:
  temperature_shift: 0
  tint_shift: 0
  exposure_offset: +0.2
  contrast: 1.15
  saturation_scale: 1.05
  vibrance_scale: 1.1

tone_curve:
  shadows_lift: 0
  highlights_roll: -5
  midtone_gamma: 1.0

split_tone:
  shadow_hue: 240                # Deep Blue
  shadow_saturation: 10
  highlight_hue: null

per_channel_hsl:
  red:    { hue: 0,   sat: +15, lum: 0 }     # Bold lips
  orange: { hue: 0,   sat: +5,  lum: +5 }
  yellow: { hue: 0,   sat: 0,   lum: 0 }
  green:  { hue: +20, sat: +10, lum: -5 }    # Emerald greens
  blue:   { hue: 0,   sat: +10, lum: 0 }
  purple: { hue: +10, sat: +15, lum: 0 }

vignette: { strength: 0, radius: 0.0 }
grain:    { amount: 0, size: 0 }
```

### 5.15 Preset: `sepia_monochrome`
*Goal: Tints and shades of a single hue (Orange/Brown).*

```yaml
global:
  temperature_shift: +10
  tint_shift: 0
  exposure_offset: 0.0
  contrast: 1.10
  saturation_scale: 0.0
  vibrance_scale: 0.0

tone_curve:
  shadows_lift: 10
  highlights_roll: -10
  midtone_gamma: 1.0

split_tone:
  shadow_hue: 30                 # Brown/Red shadows
  shadow_saturation: 30
  highlight_hue: 45              # Cream/Beige highlights
  highlight_saturation: 20

per_channel_hsl:
  red:    { hue: 0, sat: 0, lum: +10 }
  orange: { hue: 0, sat: 0, lum: +15 }
  yellow: { hue: 0, sat: 0, lum: +5 }
  green:  { hue: 0, sat: 0, lum: -10 }
  blue:   { hue: 0, sat: 0, lum: -20 }
  purple: { hue: 0, sat: 0, lum: 0 }

vignette: { strength: 0.3, radius: 0.7 }
grain:    { amount: 10, size: 1.2 }
```

### 5.16 Preset: `vibrant_landscape`
*Goal: Triadic Harmony. Pushed saturation for perceptual brilliance.*

```yaml
global:
  temperature_shift: 0
  tint_shift: 0
  exposure_offset: 0.0
  contrast: 1.15
  saturation_scale: 1.2
  vibrance_scale: 1.25

tone_curve:
  shadows_lift: 0
  highlights_roll: -15
  midtone_gamma: 1.0

split_tone:
  shadow_hue: null
  highlight_hue: null

per_channel_hsl:
  red:    { hue: 0,   sat: +10, lum: 0 }
  orange: { hue: 0,   sat: +10, lum: 0 }
  yellow: { hue: -5,  sat: +15, lum: +5 }
  green:  { hue: +5,  sat: +10, lum: -5 }
  blue:   { hue: -5,  sat: +15, lum: -10 }
  purple: { hue: 0,   sat: 0,   lum: 0 }

vignette: { strength: 0.1, radius: 0.8 }
grain:    { amount: 0, size: 1.0 }
```

### 5.17 Preset: `lavender_dream`
*Goal: Accented Analogous (Pink/Purple/Blue). Soft, ethereal.*

```yaml
global:
  temperature_shift: -2
  tint_shift: +10
  exposure_offset: +0.25
  contrast: 0.90
  saturation_scale: 0.80
  vibrance_scale: 1.0

tone_curve:
  shadows_lift: 20
  highlights_roll: -10
  midtone_gamma: 1.05

split_tone:
  shadow_hue: 260                # Lavender shadows
  shadow_saturation: 15
  highlight_hue: 330             # Pink highlights
  highlight_saturation: 10

per_channel_hsl:
  red:    { hue: +10, sat: -10, lum: +10 }
  orange: { hue: 0,   sat: 0,   lum: +5 }
  yellow: { hue: -10, sat: -20, lum: +10 }
  green:  { hue: +20, sat: -40, lum: +10 }
  blue:   { hue: +10, sat: -10, lum: +10 }
  purple: { hue: +5,  sat: +5,  lum: +5 }

vignette: { strength: 0.1, radius: 0.8 }
grain:    { amount: 5, size: 1.0 }
```

### 5.18 Preset: `bleach_bypass`
*Goal: Low Saturation, High Contrast. Gritty action feel.*

```yaml
global:
  temperature_shift: -2
  tint_shift: 0
  exposure_offset: -0.1
  contrast: 1.35
  saturation_scale: 0.40
  vibrance_scale: 0.50

tone_curve:
  shadows_lift: 0
  highlights_roll: 10
  midtone_gamma: 1.0

split_tone:
  shadow_hue: 190                # Cold Steel Cyan
  shadow_saturation: 10
  highlight_hue: 30              # Rust tone
  highlight_saturation: 10

per_channel_hsl:
  red:    { hue: 0,   sat: +10, lum: -10 }
  orange: { hue: 0,   sat: +20, lum: 0 }
  yellow: { hue: 0,   sat: -50, lum: 0 }
  green:  { hue: 0,   sat: -60, lum: -20 }
  blue:   { hue: 0,   sat: -50, lum: -10 }
  purple: { hue: 0,   sat: -50, lum: 0 }

vignette: { strength: 0.5, radius: 0.6 }
grain:    { amount: 20, size: 1.5 }
```

### 5.19 Preset: `dark_academic`
*Goal: Dyad Harmony (Green/Orange). Intellectual, dim library vibe.*

```yaml
global:
  temperature_shift: +2
  tint_shift: 0
  exposure_offset: -0.3
  contrast: 1.05
  saturation_scale: 0.90
  vibrance_scale: 1.0

tone_curve:
  shadows_lift: 10
  highlights_roll: -20
  midtone_gamma: 0.95

split_tone:
  shadow_hue: 140                # Deep Forest Green shadows
  shadow_saturation: 15
  highlight_hue: 35              # Amber/Orange highlights
  highlight_saturation: 10

per_channel_hsl:
  red:    { hue: 0,   sat: -10, lum: -5 }
  orange: { hue: -5,  sat: +5,  lum: 0 }
  yellow: { hue: -15, sat: -20, lum: -10 }
  green:  { hue: +10, sat: -10, lum: -10 }
  blue:   { hue: -10, sat: -40, lum: -20 }
  purple: { hue: 0,   sat: -40, lum: -10 }

vignette: { strength: 0.3, radius: 0.7 }
grain:    { amount: 5, size: 1.0 }
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
- hue_clamp: [15, 40]           # Force skin into healthy orange range (360Â° scale)
- saturation_limit: 65          # Cap saturation to prevent oversaturated skin
- luminance_boost: +5%          # Slight glow
- red_channel_clean: true       # Remove magenta from skin by reducing blue in red channel
- temperature_local: +3K        # Always slightly warm, even in cool presets
```

#### Sky Enhancement (Class 1)
```
- luminance_offset: -10%        # Recover highlights
- saturation_boost: +10%        # Add depth
- hue_shift: towards 195Â°       # Push generic blue â†’ teal-azure
- gradient_aware: true           # Apply more at top, less at horizon
```

#### Vegetation Taming (Class 2)
```
- hue_shift: +15Â°               # Yellow-green â†’ Emerald
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

### 6.3 Workflow: Segment â†’ Analyze â†’ Grade â†’ Blend

```
1. SEGMENT:  Run segmenter â†’ produce N binary masks
2. ANALYZE:  For each mask, compute HSL statistics (mean H, S, L)
3. GRADE:    Apply global preset (Section 5)
4. OVERRIDE: For each mask, apply per-region delta (Section 6.2)
5. BLEND:    Feather mask edges (Gaussian blur Ïƒ=5px) to avoid harsh transitions
6. MERGE:    Composite all adjusted regions back into the frame
7. FINALIZE: Apply vignette + grain from preset
```

---

## 7. Processing Order & Pipeline Integration

The color engine must execute in this exact order:

```
Input Image (RGB, 8/16-bit)
  â”‚
  â”œâ”€ 1. WHITE BALANCE CORRECTION (Global)
  â”‚     â””â”€ Grey-world or reference-based
  â”‚
  â”œâ”€ 2. EXPOSURE NORMALIZATION (Global)
  â”‚     â””â”€ Histogram stretch with soft clip
  â”‚
  â”œâ”€ 3. SEGMENTATION (SAM / SegFormer)
  â”‚     â””â”€ Generate masks for skin, sky, vegetation, dress, suit, background
  â”‚
  â”œâ”€ 4. TONE CURVE APPLICATION (Global)
  â”‚     â””â”€ shadows_lift, highlights_roll, midtone_gamma
  â”‚
  â”œâ”€ 5. CONTRAST ADJUSTMENT (Global)
  â”‚     â””â”€ S-curve in LAB L-channel
  â”‚
  â”œâ”€ 6. PER-CHANNEL HSL ADJUSTMENTS (Global then Local)
  â”‚     â”œâ”€ Apply preset per_channel_hsl globally
  â”‚     â””â”€ Apply per-region overrides using masks
  â”‚
  â”œâ”€ 7. SPLIT TONING (Global)
  â”‚     â””â”€ Colorize shadows and highlights independently
  â”‚
  â”œâ”€ 8. SATURATION / VIBRANCE (Global)
  â”‚     â””â”€ saturation_scale on all, vibrance on under-saturated pixels
  â”‚
  â”œâ”€ 9. SKIN PROTECTION PASS (Local, Mask-based)
  â”‚     â””â”€ Clamp skin hue/sat, boost luminance, clean channels
  â”‚
  â”œâ”€ 10. VIGNETTE (Global)
  â”‚     â””â”€ Radial darkening from center
  â”‚
  â””â”€ 11. GRAIN (Global)
        â””â”€ Gaussian noise overlay
```

---

## 8. Quality Assurance Checks

After grading, the pipeline validates:

| Check | Condition | Action if Failed |
|:------|:----------|:-----------------|
| Skin hue in range | H âˆˆ [15Â°, 40Â°] for skin pixels | Shift back towards 25Â° |
| Highlight clipping | < 0.5% pixels at 255 | Reduce exposure |
| Shadow clipping | < 0.5% pixels at 0 | Lift shadows |
| White dress neutral | mean(S) < 15 for dress mask | Desaturate further |
| Overall saturation | mean(S) âˆˆ [20, 75] | Scale towards center |
| Color temperature | Skin matches reference range | Re-correct WB |

