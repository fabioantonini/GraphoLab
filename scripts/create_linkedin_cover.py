"""
Genera docs/linkedin_cover.png — immagine di copertina per l'articolo LinkedIn.

Dimensioni: 1920 x 1080 px (formato raccomandato LinkedIn 2024/2025)
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "docs" / "linkedin_cover.png"

W, H = 1920, 1080

# Palette
BG_TOP    = (10,  25,  47)   # navy scuro
BG_BOT    = (17,  34,  64)   # navy medio
ACCENT    = (100, 180, 255)  # azzurro chiaro
WHITE     = (255, 255, 255)
GRAY      = (160, 180, 210)
DOT_COLOR = (30,  60, 100)


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    fonts_dir = Path("C:/Windows/Fonts")
    candidates = [
        name,
        str(fonts_dir / name),
        str(fonts_dir / "segoeui.ttf"),
        str(fonts_dir / "arial.ttf"),
        str(fonts_dir / "calibri.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def draw_gradient(draw: ImageDraw.ImageDraw) -> None:
    for y in range(H):
        t = y / H
        r = int(BG_TOP[0] * (1 - t) + BG_BOT[0] * t)
        g = int(BG_TOP[1] * (1 - t) + BG_BOT[1] * t)
        b = int(BG_TOP[2] * (1 - t) + BG_BOT[2] * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def draw_dot_pattern(draw: ImageDraw.ImageDraw) -> None:
    """Pattern di punti che richiama le impronte della scrittura."""
    spacing = 48
    for row in range(0, H // spacing + 1):
        for col in range(0, W // spacing + 1):
            cx = col * spacing + (spacing // 2 if row % 2 else 0)
            cy = row * spacing
            r = 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DOT_COLOR)


def draw_accent_lines(draw: ImageDraw.ImageDraw) -> None:
    """Linee orizzontali sottili a sinistra — richiamano righe di scrittura."""
    x0, x1 = 80, 520
    for y in [360, 380, 400, 420, 440, 460]:
        alpha = 180 if y == 360 else 80
        draw.line([(x0, y), (x1, y)], fill=(*ACCENT[:3], alpha), width=2)


def centered_text(draw: ImageDraw.ImageDraw, text: str, font, y: int,
                  fill=WHITE, x_offset: int = 0) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2 + x_offset
    draw.text((x, y), text, font=font, fill=fill)


def main() -> None:
    img  = Image.new("RGB", (W, H), BG_TOP)
    draw = ImageDraw.Draw(img)

    draw_gradient(draw)
    draw_dot_pattern(draw)

    # Linea accent verticale sinistra
    draw.rectangle([60, 200, 66, 880], fill=ACCENT)

    # Linee orizzontali "di scrittura"
    for i, y in enumerate(range(340, 560, 28)):
        length = 420 - i * 15
        opacity = max(40, 130 - i * 18)
        draw.line([(100, y), (100 + length, y)],
                  fill=(ACCENT[0], ACCENT[1], ACCENT[2]), width=1)

    # Simbolo grafico — lente stilizzata
    cx, cy, cr = 1650, 400, 120
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr],
                 outline=ACCENT, width=4)
    draw.ellipse([cx - cr + 16, cy - cr + 16, cx + cr - 16, cy + cr - 16],
                 outline=(*ACCENT, 80), width=2)
    # Manico lente
    lx = int(cx + cr * 0.7)
    ly = int(cy + cr * 0.7)
    draw.line([(lx, ly), (lx + 80, ly + 80)], fill=ACCENT, width=6)
    draw.line([(lx + 80, ly + 80), (lx + 95, ly + 95)], fill=ACCENT, width=10)

    # Righe stilizzate dentro la lente (testo simulato)
    for i, offset in enumerate([-40, -15, 10, 35]):
        length = [80, 60, 75, 45][i]
        draw.line([(cx - length // 2, cy + offset),
                   (cx + length // 2, cy + offset)],
                  fill=(*WHITE, 60), width=2)

    # Testo principale
    font_main   = load_font("segoeuib.ttf",  160)
    font_sub    = load_font("segoeui.ttf",    52)
    font_tag    = load_font("segoeui.ttf",    38)
    font_badge  = load_font("segoeui.ttf",    34)
    font_url    = load_font("segoeui.ttf",    28)

    # "GraphoLab"
    centered_text(draw, "GraphoLab", font_main, 300)

    # Linea accent sotto il titolo
    bbox = draw.textbbox((0, 0), "GraphoLab", font=font_main)
    tw = bbox[2] - bbox[0]
    lx0 = (W - tw) // 2
    draw.rectangle([lx0, 490, lx0 + tw, 496], fill=ACCENT)

    # Sottotitolo
    centered_text(draw, "Artificial Intelligence in Forensic Graphology",
                  font_sub, 530, fill=ACCENT)

    # Tag line
    centered_text(draw,
                  "Handwritten Text Recognition  ·  Signature Verification  ·  Writer Identification",
                  font_tag, 620, fill=GRAY)

    # Badge
    badge_text = "8 Labs  ·  Gradio App  ·  Open Source  ·  Apache 2.0"
    bbox = draw.textbbox((0, 0), badge_text, font=font_badge)
    bw = bbox[2] - bbox[0] + 60
    bh = bbox[3] - bbox[1] + 24
    bx = (W - bw) // 2
    by = 730
    draw.rounded_rectangle([bx, by, bx + bw, by + bh],
                            radius=8, fill=(20, 50, 90), outline=ACCENT, width=2)
    centered_text(draw, badge_text, font_badge, by + 12, fill=WHITE)

    # URL GitHub
    centered_text(draw, "github.com/fabioantonini/GraphoLab",
                  font_url, 840, fill=GRAY)

    # Linea di separazione in basso
    draw.rectangle([0, H - 6, W, H], fill=ACCENT)

    img.save(OUT, "PNG")
    print(f"Creato: {OUT}  ({W}x{H}px)")


if __name__ == "__main__":
    main()
