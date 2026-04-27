import numpy as np

def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
    """对矩阵列单位化."""
    return matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8)

def macenko_normalize(
    img: np.ndarray,
    Io: float = 240.0,
    beta: float = 0.15,
    alpha: float = 1.0,
    he_ref: np.ndarray = np.array([[0.650, 0.072], [0.704, 0.990], [0.286, 0.105]]),
    maxC_ref: np.ndarray = np.array([1.9705, 1.0308]),
) -> np.ndarray:
    """
    Macenko 染色归一化（H&E）。

    步骤：OD 转换 → SVD 求染色基 → 按目标染色基/浓度重建 RGB。
    参数:
        img: H×W×3 RGB uint8。
        Io: 光强常数（背景强度），通常 240~255。
        beta: 过滤背景的 OD 阈值。
        alpha: 角度分位数，用于确定两个主染色方向（%）。
        he_ref: 目标染色基（H&E）列向量 3×2。
        maxC_ref: 目标浓度上限（两通道）。
    返回:
        归一化后的 uint8 RGB。
    """
    img = img.astype(np.float32)
    OD = -np.log((img + 1.0) / Io)
    OD = np.clip(OD, 0.0, None)
    mask = (OD > beta).any(axis=2)
    ODhat = OD[mask].reshape(-1, 3)
    if ODhat.size == 0:
        return img.astype(np.uint8)
    _, _, V = np.linalg.svd(ODhat, full_matrices=False)
    vecs = V[:2, :].T
    proj = np.dot(ODhat, vecs)
    angles = np.arctan2(proj[:, 1], proj[:, 0])
    min_ang, max_ang = np.percentile(angles, [alpha, 100 - alpha])
    stains = np.array([[np.cos(min_ang), np.cos(max_ang)], [np.sin(min_ang), np.sin(max_ang)]])
    stain_matrix = _normalize_columns(np.dot(vecs, stains))
    C, _, _, _ = np.linalg.lstsq(stain_matrix, OD.reshape(-1, 3).T, rcond=None)
    max_C = np.percentile(C, 99, axis=1)
    scale = (maxC_ref / (max_C + 1e-8)).reshape(-1, 1)
    C_scaled = C * scale
    OD_norm = np.dot(he_ref, C_scaled)
    img_recon = Io * np.exp(-OD_norm)
    img_recon = np.clip(img_recon.T.reshape(img.shape), 0, 255).astype(np.uint8)
    return img_recon
