"""
Abstracción de la fuente de vídeo (webcam, archivo, RTSP)

Uso:
    from motion_sentinel.capture.video_source import VideoSource

    with VideoSource(source=0, width=1280, height=720) as vs:
        for frame in vs.frames():
            # frame es un numpy.ndarray BGR
            ...
"""
from collections.abc import Generator

import cv2
import numpy as np

from motion_sentinel.common.logger import get_logger

log = get_logger(__name__)


class VideoSourceError(RuntimeError):
    """Se lanza cuando la fuente de vídeo no puede abrirse o falla."""


class VideoSource:
    """
    Envuelve cv2.VideoCapture con apertura/cierre seguro y configuración
    de resolución y FPS.

    Parámetros
    ----------
    source:
        Índice de cámara (int), ruta de archivo o URL RTSP (str).
    width, height:
        Resolución solicitada (puede no ser respetada por todos los dispositivos).
    fps:
        FPS solicitados al dispositivo.
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self._cap: cv2.VideoCapture | None = None

        # True si la fuente es un archivo de vídeo local
        self._is_file = isinstance(source, str)


    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def open(self) -> "VideoSource":
        log.info("Abriendo fuente de vídeo", source=self.source)
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise VideoSourceError(f"No se puede abrir la fuente de vídeo: {self.source!r}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        log.info(
            "Fuente de vídeo abierta",
            resolution=f"{actual_w}x{actual_h}",
            fps=actual_fps,
        )
        return self


    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Fuente de vídeo cerrada")


    def __enter__(self) -> "VideoSource":
        return self.open()


    def __exit__(self, *_: object) -> None:
        self.close()


    # ------------------------------------------------------------------
    # Frame iteration
    # ------------------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """Lee un único frame. Devuelve None si la fuente se agotó."""
        if self._cap is None:
            raise VideoSourceError("VideoSource no está abierto. Llama primero a open()")

        ok, frame = self._cap.read()

        if ok:
            return frame

        # Reiniciar automáticamente archivos de vídeo
        if self._is_file:
            log.info("Fin del vídeo alcanzado, reiniciando reproducción")

            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            ok, frame = self._cap.read()

            if ok:
                return frame

        return None


    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generador que produce frames hasta que la fuente se agota."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame


    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Permite: for frame in video_source."""
        return self.frames()
