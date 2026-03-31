"""
GraphoLab core — Writer Identification.

Provides:
  - writer_identify()   identify the writer of a handwriting sample
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import filters, transform as sk_transform
from skimage.feature import hog, local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

WRITER_IMG_SIZE = (128, 256)   # (H, W) for feature extraction

_WRITER_NAMES = {
    0: "Scrittore A",
    1: "Scrittore B",
    2: "Scrittore C",
    3: "Scrittore D",
    4: "Scrittore E",
}

_FONTS_DIR = Path("C:/Windows/Fonts")
_WRITER_FONTS = [
    ("Inkfree.ttf",  19),
    ("LHANDW.TTF",   17),
    ("segoepr.ttf",  18),
    ("segoesc.ttf",  16),
    ("comic.ttf",    18),
]

_SENTENCES = [
    "il gatto dorme sul tetto",
    "la casa è piccola e bella",
    "oggi il cielo è molto blu",
    "scrivere a mano è un'arte",
    "ogni persona ha uno stile",
    "il sole tramonta a ovest",
    "leggo un libro ogni sera",
    "la penna scorre sul foglio",
    "le parole raccontano storie",
    "questo è un campione scritto",
]

# ──────────────────────────────────────────────────────────────────────────────
# Lazy model state
# ──────────────────────────────────────────────────────────────────────────────

_writer_clf: Pipeline | None = None
_writer_le: LabelEncoder | None = None
_writer_X_scaled: np.ndarray | None = None
_writer_dist_threshold: float | None = None
_writer_lock = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_synthetic_writer(writer_id: int, sample_id: int) -> Image.Image:
    """Generate a synthetic handwriting sample using system TTF fonts."""
    rng = np.random.default_rng(writer_id * 1000 + sample_id)
    font_name, base_size = _WRITER_FONTS[writer_id % len(_WRITER_FONTS)]
    font_size = base_size + int(rng.integers(-1, 2))
    try:
        font = ImageFont.truetype(str(_FONTS_DIR / font_name), font_size)
    except Exception:
        font = ImageFont.load_default()

    ink_value = int([25, 15, 35, 20, 30][writer_id % 5] + rng.integers(-5, 6))
    lines = [
        _SENTENCES[(writer_id * 3 + sample_id + i) % len(_SENTENCES)]
        for i in range(3)
    ]

    w, h = 320, 140
    img = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(img)
    line_gap = font_size + 12 + int(rng.integers(-2, 3))
    y = 10
    for line in lines:
        x = 8 + int(rng.integers(-3, 4))
        draw.text((x, y), line, fill=ink_value, font=font)
        y += line_gap

    angle = float(rng.uniform(-1.5, 1.5))
    img = img.rotate(angle, fillcolor=255, expand=False)
    return img


def _preprocess_writer_img(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to normalised grayscale array of WRITER_IMG_SIZE."""
    gray = pil_img.convert("L")
    w, h = gray.size
    target_ratio = WRITER_IMG_SIZE[1] / WRITER_IMG_SIZE[0]  # 2.0
    if h > w:
        crop_h = int(w / target_ratio)
        top = h // 6
        top = min(top, max(0, h - crop_h))
        gray = gray.crop((0, top, w, top + crop_h))
    arr = np.array(gray, dtype=np.float32)
    thresh = filters.threshold_otsu(arr) if arr.std() > 1 else 128.0
    binary = (arr < thresh).astype(np.float32)
    resized = sk_transform.resize(binary, WRITER_IMG_SIZE, anti_aliasing=True)
    return resized.astype(np.float32)


def _extract_writer_features(pil_img: Image.Image) -> np.ndarray:
    """Extract HOG + LBP + run-length features for writer identification."""
    arr = _preprocess_writer_img(pil_img)
    arr8 = (arr * 255).astype(np.uint8)

    hog_feats = hog(
        arr,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    lbp = local_binary_pattern(arr8, P=24, R=3, method="uniform")
    lbp_hist, _ = np.histogram(lbp, bins=26, range=(0, 26), density=True)

    def _run_stats(binary_row):
        runs = []
        cnt = 0
        for v in binary_row:
            if v > 0.5:
                cnt += 1
            elif cnt > 0:
                runs.append(cnt)
                cnt = 0
        if cnt > 0:
            runs.append(cnt)
        return runs

    h_runs, v_runs = [], []
    for row in arr:
        h_runs.extend(_run_stats(row))
    for col in arr.T:
        v_runs.extend(_run_stats(col))

    h_arr = np.array(h_runs, dtype=np.float32) if h_runs else np.array([0.0])
    v_arr = np.array(v_runs, dtype=np.float32) if v_runs else np.array([0.0])
    run_feats = np.array([
        h_arr.mean(), h_arr.std(), h_arr.max(),
        v_arr.mean(), v_arr.std(), v_arr.max(),
    ], dtype=np.float32)

    return np.concatenate([hog_feats, lbp_hist, run_feats])


def _load_real_writer_samples(samples_dir: Path) -> tuple[list, list] | None:
    """Load samples from data/samples/writer_XX/sample_YY.png directories."""
    writer_dirs = sorted(samples_dir.glob("writer_??"))
    if len(writer_dirs) < 2:
        return None
    X, y = [], []
    for wd in writer_dirs:
        samples = sorted(wd.glob("sample_*.png"))
        if len(samples) < 3:
            continue
        for sp in samples:
            try:
                img = Image.open(sp)
                X.append(_extract_writer_features(img))
                y.append(wd.name)
            except Exception:
                pass
    if len(set(y)) < 2:
        return None
    return X, y


def _get_writer_model(samples_dir: Path):
    """Return (Pipeline, LabelEncoder), training lazily on first call (thread-safe)."""
    global _writer_clf, _writer_le, _writer_X_scaled, _writer_dist_threshold
    if _writer_clf is not None:
        return _writer_clf, _writer_le
    with _writer_lock:
        if _writer_clf is not None:
            return _writer_clf, _writer_le
        print("Training writer identification model...")

    real = _load_real_writer_samples(samples_dir)
    if real is not None:
        X_raw, labels = real
    else:
        X_raw, labels = [], []
        for wid in range(5):
            for sid in range(10):
                img = _make_synthetic_writer(wid, sid)
                X_raw.append(_extract_writer_features(img))
                labels.append(_WRITER_NAMES[wid])

    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    X = np.array(X_raw)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=10, gamma="scale", probability=True)),
    ])
    clf.fit(X, y_enc)

    X_scaled = clf.named_steps["scaler"].transform(X)
    max_intra = 0.0
    for cls in np.unique(y_enc):
        Xc = X_scaled[y_enc == cls]
        if len(Xc) < 2:
            continue
        diff = Xc[:, np.newaxis, :] - Xc[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        max_intra = max(max_intra, dists.min(axis=1).max())

    _writer_X_scaled = X_scaled
    _writer_dist_threshold = max_intra * 2.0
    _writer_clf = clf
    _writer_le = le
    print(
        f"Writer model ready — {len(le.classes_)} writers, {len(X)} samples. "
        f"Rejection threshold: {_writer_dist_threshold:.3f}"
    )
    return _writer_clf, _writer_le


def ensure_writer_examples(examples_dir: Path) -> list[str]:
    """Pre-generate example images for UI examples."""
    examples_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for wid in range(5):
        p = examples_dir / f"writer_{wid}_example.png"
        if not p.exists():
            img = _make_synthetic_writer(wid, sample_id=99)
            img.save(str(p))
        paths.append(str(p))
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def writer_identify(image: np.ndarray, samples_dir: Path) -> tuple[str, np.ndarray | None]:
    """Identify the most likely writer of a handwriting sample.

    Args:
        image:       RGB numpy array of the handwriting sample.
        samples_dir: Path to data/samples/ directory (for real writer samples).

    Returns:
        report_md:  Markdown with ranked candidates.
        chart:      Bar chart as numpy array (or None on error).
    """
    if image is None:
        return "Carica un'immagine di testo manoscritto.", None
    try:
        clf, le = _get_writer_model(samples_dir)
    except Exception as e:
        return f"Errore nel caricamento del modello: {e}", None

    pil_img = Image.fromarray(image)
    try:
        feat = _extract_writer_features(pil_img)
    except Exception as e:
        return f"Errore nell'estrazione delle caratteristiche: {e}", None

    proba = clf.predict_proba([feat])[0]
    order = np.argsort(proba)[::-1]
    names = le.inverse_transform(order)
    scores = proba[order]

    is_unknown = False
    if _writer_X_scaled is not None and _writer_dist_threshold is not None:
        feat_scaled = clf.named_steps["scaler"].transform([feat])[0]
        min_dist = np.linalg.norm(_writer_X_scaled - feat_scaled, axis=1).min()
        is_unknown = min_dist > _writer_dist_threshold

    rows = "\n".join(
        f"| {'🥇' if i == 0 else '🥈' if i == 1 else '🥉' if i == 2 else '  '} "
        f"**{name}** | {score:.1%} |"
        for i, (name, score) in enumerate(zip(names, scores))
    )
    if is_unknown:
        report_md = (
            "**⚠️ Scrittore non identificato nel database**\n\n"
            "La scrittura analizzata non corrisponde a nessuno degli scrittori noti. "
            "Le probabilità di seguito hanno valore puramente indicativo "
            "e **non devono essere usate per un'attribuzione**.\n\n"
            "| Candidato | Probabilità (riferimento) |\n"
            "|-----------|---------------------------|\n"
            + rows
            + "\n\n*La distanza dal campione più simile nel database supera la soglia "
              "di affidabilità. Aggiungere campioni dello scrittore al database per "
              "un confronto diretto.*"
        )
    else:
        report_md = (
            "**Identificazione Scrittore — Risultati**\n\n"
            "| Candidato | Probabilità |\n"
            "|-----------|-------------|\n"
            + rows
            + "\n\n*I risultati si basano su caratteristiche HOG + LBP + statistiche dei tratti.*"
        )
    if _load_real_writer_samples(samples_dir) is None:
        report_md += (
            "\n\n⚠️ *Dati sintetici: il modello è addestrato su scritture generate "
            "artificialmente. Per risultati forensi reali, popola `data/samples/writer_XX/`.*"
        )

    # Bar chart
    fig, ax = plt.subplots(figsize=(5, max(2.5, len(names) * 0.55)))
    if is_unknown:
        colors = ["#aaaaaa"] * len(names)
        chart_title = "Scrittore non nel database — solo riferimento"
    else:
        colors = [
            "#1B3A6B" if i == 0 else "#C8973A" if i == 1 else "#9eb8e0"
            for i in range(len(names))
        ]
        chart_title = "Probabilità per scrittore"
    ax.barh(names[::-1], scores[::-1] * 100, color=colors[::-1])
    ax.set_xlabel("Probabilità (%)")
    ax.set_xlim(0, 105)
    ax.set_title(chart_title)
    for i, (name, score) in enumerate(zip(names[::-1], scores[::-1])):
        ax.text(score * 100 + 1, i, f"{score:.1%}", va="center", fontsize=9)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    chart_arr = np.array(Image.open(buf))

    return report_md, chart_arr
