"""
Crea 3 documenti fittizi con date in formato italiano per testare
il tab "Datazione Documenti" di GraphoLab.

Output:
  data/samples/doc_lettera_1998.png      — data: 3 marzo 1998
  data/samples/doc_contratto_2001.png    — data: 15/06/2001
  data/samples/doc_testamento_2024.png   — data: 10 gennaio 2024

Ordinamento atteso nella tab: 1998 → 2001 → 2024
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "samples"

FONTS_DIR = Path("C:/Windows/Fonts")
PAGE_W, PAGE_H = 1200, 900
MARGIN = 80
BG     = (255, 255, 255)
INK    = (30, 30, 30)
GRAY   = (120, 120, 120)


def load_font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in [name, str(FONTS_DIR / name)]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img  = Image.new("RGB", (PAGE_W, PAGE_H), BG)
    draw = ImageDraw.Draw(img)
    # bordo sottile
    draw.rectangle([10, 10, PAGE_W - 11, PAGE_H - 11], outline=(200, 200, 200), width=2)
    return img, draw


def draw_text_block(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
                    x: int, y: int, max_w: int, fill=INK, align="left") -> int:
    """Disegna testo con a capo automatico. Restituisce la y finale."""
    words  = text.split()
    line   = ""
    lines  = []
    for word in words:
        test = (line + " " + word).strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_w and line:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)

    line_h = font.size + 6
    for ln in lines:
        if align == "right":
            bbox = draw.textbbox((0, 0), ln, font=font)
            lw   = bbox[2] - bbox[0]
            draw.text((x + max_w - lw, y), ln, font=font, fill=fill)
        else:
            draw.text((x, y), ln, font=font, fill=fill)
        y += line_h
    return y


# ──────────────────────────────────────────────
# Documento 1: Lettera formale — 3 marzo 1998
# ──────────────────────────────────────────────
def create_lettera(path: Path) -> None:
    img, draw = new_canvas()
    font_h  = load_font("Inkfree.ttf",  28)
    font_b  = load_font("Inkfree.ttf",  20)
    font_sm = load_font("Inkfree.ttf",  17)
    font_dt = load_font("Inkfree.ttf",  22)

    x0   = MARGIN
    max_w = PAGE_W - 2 * MARGIN

    # Data in alto a destra
    draw.text((PAGE_W - MARGIN, MARGIN), "3 marzo 1998",
              font=font_dt, fill=INK, anchor="ra")

    # Mittente
    y = MARGIN
    draw.text((x0, y), "Dott. Mario Bianchi", font=font_b, fill=INK)
    y += 28
    draw.text((x0, y), "Via Roma 14, 20121 Milano", font=font_sm, fill=GRAY)
    y += 70

    # Intestazione
    draw.text((x0, y), "LETTERA RACCOMANDATA", font=font_h, fill=INK)
    y += 50

    draw.line([(x0, y), (PAGE_W - MARGIN, y)], fill=(180, 180, 180), width=1)
    y += 20

    # Destinatario
    draw.text((x0, y), "Egregio Avv. Luigi Ferri", font=font_b, fill=INK)
    y += 28
    draw.text((x0, y), "Studio Legale Ferri & Associati", font=font_sm, fill=GRAY)
    y += 28
    draw.text((x0, y), "Corso Vittorio Emanuele 88, 00186 Roma", font=font_sm, fill=GRAY)
    y += 50

    # Corpo lettera
    corpo = (
        "Con la presente mi pregio di informarLa che, in seguito alla nostra riunione "
        "del mese scorso, ho provveduto a raccogliere tutta la documentazione necessaria "
        "relativa alla pratica in oggetto. Allego pertanto le copie dei contratti stipulati "
        "nonché le ricevute di pagamento degli ultimi tre anni fiscali. "
        "Resto a Sua completa disposizione per qualsiasi chiarimento."
    )
    y = draw_text_block(draw, corpo, font_sm, x0, y, max_w)
    y += 30

    draw.text((x0, y), "Distinti saluti,", font=font_sm, fill=INK)
    y += 30
    draw.text((x0, y), "Mario Bianchi", font=font_b, fill=INK)
    y += 28
    draw.line([(x0, y), (x0 + 200, y)], fill=INK, width=1)

    img.save(path, "PNG")
    print(f"  Creato: {path.name}")


# ──────────────────────────────────────────────
# Documento 2: Contratto di vendita — 15/06/2001
# ──────────────────────────────────────────────
def create_contratto(path: Path) -> None:
    img, draw = new_canvas()
    font_h  = load_font("segoesc.ttf",  26)
    font_b  = load_font("segoesc.ttf",  19)
    font_sm = load_font("segoesc.ttf",  16)
    font_dt = load_font("segoesc.ttf",  21)

    x0    = MARGIN
    max_w = PAGE_W - 2 * MARGIN

    y = MARGIN
    # Titolo centrato
    title = "CONTRATTO DI COMPRAVENDITA IMMOBILIARE"
    bbox  = draw.textbbox((0, 0), title, font=font_h)
    tw    = bbox[2] - bbox[0]
    draw.text(((PAGE_W - tw) // 2, y), title, font=font_h, fill=INK)
    y += 50

    # Data e luogo
    draw.text((x0, y), "Luogo e data: Torino, 15/06/2001", font=font_dt, fill=INK)
    y += 45

    draw.line([(x0, y), (PAGE_W - MARGIN, y)], fill=(160, 160, 160), width=2)
    y += 25

    # Parti contraenti
    draw.text((x0, y), "TRA", font=font_b, fill=GRAY)
    y += 28
    draw.text((x0, y), "Sig. Carlo Esposito (venditore), C.F. ESPCRL60M15F205X", font=font_sm, fill=INK)
    y += 25
    draw.text((x0, y), "E", font=font_b, fill=GRAY)
    y += 28
    draw.text((x0, y), "Sig.ra Anna De Luca (acquirente), C.F. DLCNNA72E55L219Q", font=font_sm, fill=INK)
    y += 40

    # Articoli
    articoli = [
        ("Art. 1 — Oggetto", "Il venditore cede all'acquirente l'immobile sito in Via Garibaldi 5, Torino, "
         "catastalmente identificato al Foglio 12, Particella 340, sub 8."),
        ("Art. 2 — Prezzo", "Il prezzo pattuito ammonta a lire 280.000.000 (duecentoottantamilioni), "
         "già interamente corrisposto alla data di firma del presente atto."),
        ("Art. 3 — Consegna", "La consegna delle chiavi avverrà entro trenta giorni dalla firma, "
         "libero da persone e cose."),
    ]
    for titolo, testo in articoli:
        draw.text((x0, y), titolo, font=font_b, fill=INK)
        y += 26
        y = draw_text_block(draw, testo, font_sm, x0 + 20, y, max_w - 20)
        y += 15

    y += 20
    col1, col2 = x0, PAGE_W // 2
    draw.text((col1, y), "Il venditore", font=font_sm, fill=GRAY)
    draw.text((col2, y), "L'acquirente", font=font_sm, fill=GRAY)
    y += 50
    draw.line([(col1, y), (col1 + 200, y)], fill=INK, width=1)
    draw.line([(col2, y), (col2 + 200, y)], fill=INK, width=1)

    img.save(path, "PNG")
    print(f"  Creato: {path.name}")


# ──────────────────────────────────────────────
# Documento 3: Testamento olografo — 10 gennaio 2024
# ──────────────────────────────────────────────
def create_testamento(path: Path) -> None:
    img, draw = new_canvas()
    font_h  = load_font("segoepr.ttf",  26)
    font_b  = load_font("segoepr.ttf",  20)
    font_sm = load_font("segoepr.ttf",  17)
    font_dt = load_font("segoepr.ttf",  20)

    x0    = MARGIN
    max_w = PAGE_W - 2 * MARGIN

    y = MARGIN
    # Titolo
    title = "TESTAMENTO OLOGRAFO"
    bbox  = draw.textbbox((0, 0), title, font=font_h)
    tw    = bbox[2] - bbox[0]
    draw.text(((PAGE_W - tw) // 2, y), title, font=font_h, fill=INK)
    y += 55

    draw.line([(x0, y), (PAGE_W - MARGIN, y)], fill=(180, 180, 180), width=1)
    y += 25

    # Testatore
    draw.text((x0, y), "Io sottoscritta, Giovanna Mancini, nata a Napoli il 12 aprile 1942,", font=font_sm, fill=INK)
    y += 28
    draw.text((x0, y), "residente in Via Caracciolo 22, 80122 Napoli, nel pieno possesso", font=font_sm, fill=INK)
    y += 28
    draw.text((x0, y), "delle mie facoltà mentali, dispongo delle mie ultime volontà:", font=font_sm, fill=INK)
    y += 45

    # Disposizioni
    disposizioni = [
        "1. Lascio al mio figlio primogenito, Dott. Roberto Mancini, "
        "l'appartamento di Via Caracciolo 22 e tutti i beni mobili ivi contenuti.",
        "2. Lascio alla mia nipote Sara Mancini la somma di euro 50.000 (cinquantamila) "
        "depositata sul conto corrente n. 123456 presso la Banca di Napoli.",
        "3. Il residuo del mio patrimonio dovrà essere suddiviso in parti uguali "
        "tra i miei tre figli: Roberto, Elena e Marco Mancini.",
        "4. Nomino esecutore testamentario il Notaio Dott. Enzo Vitale, "
        "con studio in Via Toledo 80, Napoli.",
    ]
    for d in disposizioni:
        y = draw_text_block(draw, d, font_sm, x0, y, max_w)
        y += 18

    y += 30
    # Data in basso — font più grande e left-aligned per facilitare l'OCR
    font_dt2 = load_font("segoepr.ttf", 26)
    draw.text((x0, y), "Napoli, 10 gennaio 2024", font=font_dt2, fill=INK)
    y += 50

    draw.text((x0, y), "Firma autografa:", font=font_sm, fill=GRAY)
    y += 35
    draw.line([(x0, y), (x0 + 250, y)], fill=INK, width=1)
    y += 10
    draw.text((x0, y), "Giovanna Mancini", font=font_b, fill=INK)

    img.save(path, "PNG")
    print(f"  Creato: {path.name}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generazione documenti fittizi con date...")
    create_lettera(OUT_DIR / "doc_lettera_1998.png")
    create_contratto(OUT_DIR / "doc_contratto_2001.png")
    create_testamento(OUT_DIR / "doc_testamento_2024.png")
    print("Fatto. Carica i 3 file nel tab 'Datazione Documenti' di GraphoLab.")
