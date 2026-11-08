"""Procedural textures for the v3 scene: a small set of wood-grain table tops and sticker-cube faces."""
import numpy as np
import cv2

WOOD_PALETTES = {   # (dark, light) RGB 0..1; first entry is closest to the owner's bench (dark stained planks)
    "walnut_stain": ((0.20, 0.11, 0.07), (0.50, 0.30, 0.19)),
    "walnut": ((0.26, 0.16, 0.10), (0.55, 0.36, 0.23)),
    "oak": ((0.52, 0.38, 0.22), (0.80, 0.64, 0.42)),
    "pine": ((0.66, 0.50, 0.30), (0.90, 0.78, 0.56)),
    "cherry": ((0.38, 0.17, 0.10), (0.66, 0.36, 0.22)),
    "grey_wash": ((0.36, 0.34, 0.32), (0.66, 0.63, 0.59)),
    "ebony": ((0.10, 0.08, 0.07), (0.30, 0.23, 0.18)),
    "birch": ((0.70, 0.62, 0.48), (0.93, 0.88, 0.76)),
}
STICKERS = np.array([[0.95, 0.95, 0.95], [0.98, 0.84, 0.05], [0.80, 0.07, 0.10], [0.98, 0.42, 0.04], [0.05, 0.27, 0.72], [0.02, 0.58, 0.25]])


def _noise(rng, shape, scale_yx):
    small = rng.standard_normal((max(2, shape[0] // scale_yx[0]), max(2, shape[1] // scale_yx[1]))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def wood(seed, palette="walnut_stain", size=2048, table_m=1.5):
    """Planks along the image x axis with seams, long grain streaks, stains and knots. uint8 (size, size, 3)."""
    rng = np.random.default_rng(seed)
    dark, light = (np.array(c, np.float32) for c in WOOD_PALETTES[palette])
    px_per_m = size / table_m
    tone = np.zeros((size, size), np.float32); seam = np.zeros((size, size), np.float32)
    y = 0
    while y < size:
        width = int(rng.uniform(0.085, 0.16) * px_per_m); y1 = min(size, y + width)
        tone[y:y1] = rng.uniform(0.25, 0.75)
        tone[y:y1] += 0.10 * np.sin(np.linspace(0, rng.uniform(2, 9), size) + rng.uniform(0, 6))[None, :]
        seam[max(0, y - 1):y + 2] = 1.0
        for _ in range(rng.integers(0, 3)):          # butt joints along the plank
            x = rng.integers(0, size); seam[y:y1, x:x + 2] = 1.0
        y = y1
    grain = 0.55 * _noise(rng, (size, size), (3, 160)) + 0.30 * _noise(rng, (size, size), (2, 48)) + 0.15 * _noise(rng, (size, size), (1, 6))
    stain = np.clip(_noise(rng, (size, size), (180, 260)) - 0.4, 0, None)
    value = np.clip(tone + 0.16 * grain - 0.35 * stain, 0, 1)
    for _ in range(rng.integers(2, 9)):               # knots
        cx, cy, r = rng.integers(0, size), rng.integers(0, size), rng.uniform(6, 22)
        yy, xx = np.ogrid[:size, :size]; d = np.sqrt(((xx - cx) / 1.8) ** 2 + (yy - cy) ** 2)
        value -= 0.45 * np.exp(-(d / r) ** 2)
    value = np.clip(value, 0, 1)[..., None]
    rgb = dark + (light - dark) * value
    rgb *= (1 - 0.65 * cv2.GaussianBlur(seam, (0, 0), 1.2))[..., None]
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def rubik_faces(seed, sticker=96, hue_jitter=0.06, scheme="scrambled"):
    """Six faces stacked vertically (MuJoCo cube-map layout), each 3x3 stickers on a black body. uint8 (18*s, 3*s, 3)."""
    rng = np.random.default_rng(seed)
    colours = np.clip(STICKERS + rng.uniform(-hue_jitter, hue_jitter, STICKERS.shape), 0, 1)
    if scheme == "recolour": colours = rng.uniform(0.05, 0.95, (6, 3))
    body = np.array([0.03, 0.03, 0.03]) + rng.uniform(0, 0.05)
    faces = []
    for face in range(6):
        img = np.ones((3 * sticker, 3 * sticker, 3), np.float32) * body
        solved = rng.random() < 0.25
        for i in range(3):
            for j in range(3):
                colour = colours[face] if solved else colours[rng.integers(0, 6)]
                pad = max(3, sticker // 12); y0, x0 = i * sticker + pad, j * sticker + pad
                cv2.rectangle(img, (x0, y0), (x0 + sticker - 2 * pad, y0 + sticker - 2 * pad), tuple(float(c) for c in colour), -1, cv2.LINE_AA)
        faces.append(img)
    return np.clip(np.concatenate(faces, 0) * 255, 0, 255).astype(np.uint8)
