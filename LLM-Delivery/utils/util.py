import numpy as np
from PIL import Image
from io import BytesIO


# --- 仅处理两类：bytes 和 ndarray ---
def _ensure_png_bytes(img) -> bytes:
    # 1) 已是 bytes：直返
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)

    # 2) ndarray：转 PNG bytes（支持灰度/BGR/BGRA/float）

    arr = img
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            mx = float(arr.max()) if arr.size else 1.0
            if mx <= 1.0:
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0.0, 255.0).round().astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        mode = "L"
        out = arr
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # BGR -> RGB
        out = arr[:, :, ::-1].copy()
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        # BGRA -> RGBA（保留 alpha）
        b, g, r, a = arr.transpose(2, 0, 1)
        out = np.dstack([r, g, b, a])
        mode = "RGBA"
    else:
        raise ValueError(f"unsupported ndarray shape: {arr.shape}")

    out = np.ascontiguousarray(out)
    bio = BytesIO()
    Image.fromarray(out, mode=mode).save(bio, format="PNG")
    return bio.getvalue()

