"""
Funciones de preprocesamiento de frames para motion-sentinel.

Todas las funciones reciben y devuelven numpy.ndarray (formato OpenCV).
No modifican el frame original (operaciones sin in-place).
"""

import cv2
import numpy as np


def resize_frame(frame: np.ndarray, width: int | None = None) -> np.ndarray:
    """
    Redimensiona el frame manteniendo la relación de aspecto.

    Si `width` es None o el frame ya tiene ese ancho, devuelve el frame sin
    cambios. El alto se calcula automáticamente para evitar distorsión.
    """
    if width is None:
        return frame

    current_h, current_w = frame.shape[:2]
    if current_w == width:
        return frame

    scale = width / current_w
    new_h = int(current_h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convierte un frame BGR a escala de grises (canal único).

    Si el frame ya es de un solo canal (grayscale), lo devuelve tal cual.
    """
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(frame: np.ndarray, kernel_size: int = 21) -> np.ndarray:
    """
    Aplica desenfoque gaussiano para reducir ruido de alta frecuencia.

    Un kernel más grande suaviza más pero pierde detalle fino.
    El kernel debe ser impar; si se pasa un valor par se ajusta al siguiente impar.
    """
    kernel_size = _ensure_odd(kernel_size)
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigmaX=0)


def preprocess_for_motion(
    frame: np.ndarray,
    resize_width: int | None = None,
    blur_kernel: int = 21,
) -> np.ndarray:
    """
    Pipeline completo de preprocesamiento para detección de movimiento.

    Pasos en orden:
        1. Redimensionar (opcional) — reduce carga computacional.
        2. Convertir a escala de grises — un canal es suficiente para motion.
        3. Aplicar desenfoque gaussiano — atenúa el ruido antes de comparar frames.

    Devuelve un array 2-D (H, W) uint8 listo para pasarse al detector de movimiento.
    """
    processed = resize_frame(frame, width=resize_width)
    processed = to_grayscale(processed)
    processed = apply_gaussian_blur(processed, kernel_size=blur_kernel)
    return processed


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _ensure_odd(value: int) -> int:
    """Devuelve `value` si es impar, o `value + 1` si es par. Mínimo 1."""
    value = max(1, value)
    return value if value % 2 == 1 else value + 1
