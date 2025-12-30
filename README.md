# Dress Color Changer

Interactive tool to change colors in images using HLS color space transformations.

## Requirements

```bash
pip install numpy opencv-python
```

## Usage

```bash
python dress_color_changer.py -i image.jpg
```

## Controls

- **Click** on image to select target color
- **New Hue / New Sat**: Set replacement color
- **Tolerances**: Adjust selection sensitivity
- **Show Mask**: Preview selected pixels (green)
- **Q**: Quit

## How It Works

### Color Space Conversion (RGB ↔ HLS)

The program uses HLS color space where:
- **H** (Hue): 0-360° color wheel position
- **L** (Lightness): 0-1 brightness
- **S** (Saturation): 0-1 color intensity

**RGB to HLS formulas:**

```
Cmax = max(R, G, B)
Cmin = min(R, G, B)
Δ = Cmax - Cmin

L = (Cmax + Cmin) / 2

S = Δ / (1 - |2L - 1|)

H = 60° × ((G-B)/Δ mod 6)  when Cmax = R
H = 60° × ((B-R)/Δ + 2)    when Cmax = G
H = 60° × ((R-G)/Δ + 4)    when Cmax = B
```

**HLS to RGB formulas:**

```
C = (1 - |2L - 1|) × S
X = C × (1 - |H/60 mod 2 - 1|)
m = L - C/2

Then RGB values based on H sector + m offset
```

### Color Selection

Pixels are selected using distance thresholds:

```
Hue distance (circular): min(|H1-H2|, 360-|H1-H2|)
Lightness distance: |L1 - L2|
Saturation distance: |S1 - S2|
```

### Color Replacement

Only H and S channels are modified; L is preserved to keep original lighting/texture.

## Author

[Your Name] - Image Processing Course, December 2024
