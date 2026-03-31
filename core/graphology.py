"""
GraphoLab core — Graphological Feature Analysis.

Provides:
  - grapho_analyse()   extract graphological metrics from a handwriting image
"""

from __future__ import annotations

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────────────────────

def grapho_analyse(image: np.ndarray) -> tuple[str, np.ndarray]:
    """Analyse graphological features of a handwritten image.

    Args:
        image: RGB numpy array (H, W, 3).

    Returns:
        report_md:  Markdown table with extracted metrics.
        annotated:  Annotated image (bounding boxes on letters) as numpy array.
    """
    if image is None:
        return "Carica un'immagine di scrittura a mano.", image

    # Cap to 800 px: adaptive threshold is O(pixels × blockSize)
    h0, w0 = image.shape[:2]
    if max(h0, w0) > 800:
        sc = 800 / max(h0, w0)
        image = cv2.resize(image, (int(w0 * sc), int(h0 * sc)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10,
    )

    # Slant
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= 20 and len(cnt) >= 5:
            _, _, angle = cv2.fitEllipse(cnt)
            slant = angle - 90.0
            if -60 < slant < 60:
                angles.append(slant)
    slant_mean = float(np.mean(angles)) if angles else 0.0
    slant_std = float(np.std(angles)) if angles else 0.0

    # Pressure
    ink_mask = binary > 0
    pressure = (255 - gray)[ink_mask]
    pressure_mean = float(pressure.mean()) if len(pressure) else 0.0

    # Connected components
    num, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    valid = stats[1:][stats[1:, cv2.CC_STAT_AREA] > 15] if num > 1 else np.zeros((0, 5))
    h_mean = float(valid[:, cv2.CC_STAT_HEIGHT].mean()) if len(valid) else 0.0
    w_mean = float(valid[:, cv2.CC_STAT_WIDTH].mean()) if len(valid) else 0.0

    # Word spacing
    h_proj = binary.sum(axis=0)
    gaps = []
    in_gap, gap_w = False, 0
    for v in h_proj:
        if v == 0:
            in_gap = True
            gap_w += 1
        elif in_gap:
            if gap_w > 5:
                gaps.append(gap_w)
            in_gap = False
            gap_w = 0
    word_spacing = float(np.mean(gaps)) if gaps else 0.0

    ink_density = ink_mask.mean() * 100

    # Annotated visualisation
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    for cnt in contours:
        if cv2.contourArea(cnt) >= 20:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 180, 255), 1)

    slant_dir = "destra" if slant_mean > 0 else ("sinistra" if slant_mean < 0 else "verticale")
    report_md = (
        f"**Analisi delle Caratteristiche Grafologiche**\n\n"
        f"| Caratteristica | Valore |\n"
        f"|----------------|--------|\n"
        f"| Inclinazione media lettere | {slant_mean:+.1f}° ({slant_dir}) |\n"
        f"| Variazione inclinazione (σ) | {slant_std:.1f}° |\n"
        f"| Pressione del tratto | {pressure_mean:.1f} / 255 |\n"
        f"| Altezza media lettere | {h_mean:.1f} px |\n"
        f"| Larghezza media lettere | {w_mean:.1f} px |\n"
        f"| Spaziatura media parole | {word_spacing:.1f} px |\n"
        f"| Densità inchiostro | {ink_density:.2f}% |\n"
        f"| Componenti connesse | {len(valid)} |\n\n"
        f"*I bounding box delle lettere sono visibili nell'immagine annotata.*"
    )
    return report_md, vis
