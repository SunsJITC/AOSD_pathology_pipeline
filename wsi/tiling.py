import os
import numpy as np, cv2, tifffile as tiff

# Optional OpenSlide support for NDPI/SVS; code remains functional without it.
try:
    import openslide  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openslide = None

def _read_with_openslide(path: str, level: int = 0) -> np.ndarray:
    if openslide is None:
        raise RuntimeError("openslide-python is required for NDPI/SVS reading but is not installed.")
    slide = openslide.OpenSlide(path)
    w, h = slide.level_dimensions[level]
    region = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    return np.array(region)

def _read_with_tifffile(path: str, level: int = 0) -> np.ndarray:
    # Prefer pyramid-aware reading for multi-resolution TIFF.
    arr = None
    try:
        with tiff.TiffFile(path) as tf:
            if len(tf.series) > 0:
                s0 = tf.series[0]
                if hasattr(s0, "levels") and len(s0.levels) > 0:
                    lvl = max(0, min(level, len(s0.levels) - 1))
                    arr = s0.levels[lvl].asarray()
    except Exception:
        arr = None
    if arr is None:
        arr = tiff.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr

def load_rgb(path: str, level: int = 0) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ndpi", ".svs"]:
        return _read_with_openslide(path, level=level)
    return _read_with_tifffile(path, level=level)

def tissue_mask(rgb: np.ndarray):
    # LAB-based background suppression + Otsu to better separate tissue from slide.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0]
    # Otsu threshold on lightness
    _, th = cv2.threshold(L, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (th == 0).astype(np.uint8)  # tissue tends to be darker
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    return mask

def variance_of_laplacian(image: np.ndarray) -> float:
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()

def tile_image(rgb: np.ndarray, patch_size=384, overlap=0, min_tissue_percent=0.5, blur_var_th=60.0, max_tiles=6000):
    H, W, _ = rgb.shape
    stride = max(1, patch_size - overlap)
    tiles, coords = [], []
    mask = tissue_mask(rgb)
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            m = mask[y:y+patch_size, x:x+patch_size]
            if (m>0.5).mean() < min_tissue_percent:
                continue
            patch = rgb[y:y+patch_size, x:x+patch_size]
            if variance_of_laplacian(patch) < blur_var_th:
                continue
            tiles.append(patch); coords.append((x,y))
            if len(tiles) >= max_tiles:
                break
        if len(tiles) >= max_tiles:
            break
    return tiles, coords

def make_thumbnail(rgb: np.ndarray, max_size=2048):
    H, W = rgb.shape[:2]
    scale = min(max_size/max(H,W), 1.0)
    th = cv2.resize(rgb, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    return th, scale
