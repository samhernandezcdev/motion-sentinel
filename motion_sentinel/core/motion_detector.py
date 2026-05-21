"""
Detección de movimiento por diferencia absoluta entre frames consecutivos.

Pipeline:
    frame preprocesado → absdiff → threshold → dilate → contornos → MotionRegion

Solo implementa frame differencing clásico. MOG2 / KNN se agregan en fases
posteriores como estrategias intercambiables (Strategy pattern).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionRegion:
    """
    Región del frame donde se detectó movimiento.

    Atributos
    ----------
    bbox:
        Tupla (x, y, w, h) en píxeles del rectángulo delimitador.
    area:
        Área del contorno en píxeles².
    """

    bbox: tuple[int, int, int, int]
    area: float


class FrameDifferenceMotionDetector:
    """
    Detector de movimiento basado en diferencia absoluta entre frames.

    Parámetros
    ----------
    threshold:
        Umbral de intensidad (0-255) para binarizar el absdiff.
        Valores bajos → más sensible al ruido.
        Valores altos → solo diferencias marcadas se detectan.
    min_area:
        Área mínima en píxeles² para considerar un contorno como movimiento real.
        Filtra ruido residual tras el threshold.
    dilation_iterations:
        Número de iteraciones de dilatación morfológica.
        Une regiones cercanas en un solo contorno coherente.
    """

    # Kernel cuadrado 3×3 para la dilatación morfológica.
    _DILATION_KERNEL: np.ndarray = np.ones((3, 3), dtype=np.uint8)

    def __init__(
        self,
        threshold: int = 25,
        min_area: float = 500.0,
        dilation_iterations: int = 2,
    ) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.dilation_iterations = dilation_iterations

        self._prev_frame: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, current_frame: np.ndarray
    ) -> tuple[np.ndarray, list[MotionRegion]]:
        """
        Detecta movimiento comparando `current_frame` con el frame anterior.

        Espera un array 2-D (H, W) uint8 ya preprocesado (grayscale + blur).
        En la primera llamada solo almacena la referencia y devuelve una
        máscara vacía sin regiones, ya que no hay frame previo con qué comparar.

        Devuelve
        --------
        mask:
            Máscara binaria (H, W) uint8 con 255 en zonas de movimiento.
        regions:
            Lista de :class:`MotionRegion` ordenada de mayor a menor área.
        """
        empty_mask = np.zeros_like(current_frame)

        if self._prev_frame is None:
            self._prev_frame = current_frame.copy()
            return empty_mask, []

        # 1. Diferencia pixel a pixel
        diff = cv2.absdiff(self._prev_frame, current_frame)

        # 2. Umbralización binaria -> zonas con cambio significativo en blanco
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # 3. Dilatación morfológica -> une regiones fragmentados por ruido
        mask = cv2.dilate(
            mask, self._DILATION_KERNEL, iterations=self.dilation_iterations
        )

        # 4. Contornos externos sobre la máscara dilatada
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # 5. Filtrar por área mínima y construir MotionRegion
        regions: list[MotionRegion] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            regions.append(MotionRegion(bbox=(x, y, w, h), area=area))

        # Ordenar de mayor a menor área: la región más grande primero
        regions.sort(key=lambda r: r.area, reverse=True)

        # 6. Actualizar referencia para la siguiente llamada
        self._prev_frame = current_frame.copy()

        return mask, regions

    def reset(self) -> None:
        """
        Descarta el frame de referencia.

        Útil al cambiar de fuente de vídeo, al reanudar tras una pausa,
        o en tests para reiniciar el estado interno sin crear una nueva instancia.
        """
        self._prev_frame = None
