"""
Crea data/samples/testamento_writer00.png

Documento composto da campioni reali di grafia writer_00 (opzione A).
Corpo testo: campioni writer_00/sample_*.png impilati verticalmente.
Firma: CEDAR original_1_1.png (writer_00) + etichetta "Luca Rossi".
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT    = Path(__file__).parent.parent
SAMPLES = ROOT / "data/samples/writer_00"
FIRMA   = ROOT / "data/cedar/signatures/full_org/original_1_1.png"
OUT_IMG = ROOT / "data/samples/testamento_writer00.png"

# Campioni selezionati (indici diversi = frasi diverse)
SELECTED = [0, 4, 8, 12, 18, 25]

PAGE_W      = 1800
MARGIN_X    = 130
MARGIN_TOP  = 100
GAP         = 45     # spazio verticale tra campioni
SIG_GAP     = 90     # spazio prima della firma
SIG_H       = 170    # altezza target firma
MARGIN_BOT  = 100


def load_sample(idx: int, target_w: int) -> Image.Image:
    path = SAMPLES / f"sample_{idx:03d}.png"
    img  = Image.open(path).convert("L")
    scale = target_w / img.width
    new_h = int(img.height * scale)
    img   = img.resize((target_w, new_h), Image.LANCZOS)
    rgb   = img.convert("RGB")
    # assicura sfondo bianco
    return rgb


def main() -> None:
    available = sorted(SAMPLES.glob("sample_*.png"))
    max_idx   = len(available) - 1
    indices   = [i for i in SELECTED if i <= max_idx]

    text_w  = PAGE_W - 2 * MARGIN_X
    samples = [load_sample(i, text_w) for i in indices]

    # Altezza canvas
    body_h  = sum(s.height for s in samples) + GAP * (len(samples) - 1)
    page_h  = MARGIN_TOP + body_h + SIG_GAP + SIG_H + MARGIN_BOT

    canvas  = Image.new("RGB", (PAGE_W, page_h), (255, 255, 255))

    # Incolla campioni
    y = MARGIN_TOP
    for samp in samples:
        canvas.paste(samp, (MARGIN_X, y))
        y += samp.height + GAP

    # Firma CEDAR writer_1
    sig = Image.open(FIRMA).convert("RGBA")
    scale  = SIG_H / sig.height
    sig_w  = int(sig.width * scale)
    sig    = sig.resize((sig_w, SIG_H), Image.LANCZOS)

    sig_bg = Image.new("RGB", sig.size, (255, 255, 255))
    sig_bg.paste(sig, mask=sig.split()[3])

    sig_x = PAGE_W - sig_w - MARGIN_X
    sig_y = y + SIG_GAP
    canvas.paste(sig_bg, (sig_x, sig_y))

    # Etichetta "Luca Rossi"
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("inkfree.ttf", size=46)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/inkfree.ttf", size=46)
        except (IOError, OSError):
            font = ImageFont.load_default()

    lx = sig_x + sig_w // 2
    ly = sig_y + SIG_H + 8
    draw.text((lx, ly), "Luca Rossi", fill=(50, 50, 50), font=font, anchor="mt")

    canvas.save(OUT_IMG, "PNG")
    print(f"Creato: {OUT_IMG}  ({PAGE_W}x{page_h}px)")


if __name__ == "__main__":
    main()
