"""
Configuración tipada e inmutable para la fuente de vídeo.

Centraliza todos los parámetros que ``VideoSource`` necesita en un único
objeto que puede construirse desde YAML, pasarse entre funciones y
compararse en tests sin depender de ``ConfigManager``.

Lo que NO hace (fases futuras)
-------------------------------
- Configuración de ``FrameBuffer``  → Fase 2: ``FrameBufferConfig`` separado,
  compuesto junto a ``CaptureConfig`` en un ``PipelineConfig`` de nivel superior.
- Reconexión automática a RTSP      → Fase 2: campo ``reconnect_delay_s: float``
  y ``max_reconnect_attempts: int`` leídos aquí, consumidos por ``VideoSource``.
- Autenticación de streams IP       → Fase 2: campo ``credentials`` opcional,
  inyectado en la URL antes de pasarla a ``cv2.VideoCapture``.
- Perfiles de resolución            → Fase 2: ``profiles/edge.yaml`` puede
  sobrescribir ``width`` / ``height`` sin tocar la lógica del pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from motion_sentinel.common.url_utils import redact_url_credentials

if TYPE_CHECKING:
    from motion_sentinel.common.config_manager import ConfigManager


@dataclass(frozen=True)
class CaptureConfig:
    """
    Parámetros de captura de vídeo.

    Immutable por diseño (``frozen=True``): una instancia creada al arrancar
    el pipeline no puede mutarse accidentalmente durante el loop de captura.

    Atributos
    ----------
    source : int | str
        Fuente de vídeo. Acepta:
        - ``int``   — índice de webcam local (p.ej. ``0`` para la cámara por defecto).
        - ``str``   — ruta a un archivo de vídeo (``"data/test_video.mp4"``).
        - ``str``   — URL RTSP  (``"rtsp://192.168.1.10:554/stream"``).
        - ``str``   — URL MJPEG (``"http://192.168.1.10:8080/video"``).
    width : int
        Ancho solicitado al driver de captura en píxeles.
        El driver puede ignorarlo si el hardware no lo soporta.
    height : int
        Alto solicitado al driver de captura en píxeles.
    fps : int
        Framerate solicitado al driver. Para archivos y streams se usa como
        límite de lectura, no como garantía.
    window_title : str
        Título de la ventana de ``cv2.imshow``. No afecta al pipeline de
        procesamiento; es estrictamente presentacional.
    """

    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    window_title: str = "Motion Sentinel"

    @classmethod
    def from_config(cls, cfg: ConfigManager) -> CaptureConfig:
        """
        Construye una ``CaptureConfig`` leyendo valores desde ``ConfigManager``.

        Normalización de ``source``
        ---------------------------
        El YAML puede almacenar el índice de webcam como entero (``source: 0``)
        o como cadena numérica (``source: "0"``). Ambas formas producen
        ``source=0`` (``int``). Cualquier otra cadena —ruta, RTSP, MJPEG— se
        mantiene como ``str`` sin modificación.

        Parámetros
        ----------
        cfg : ConfigManager
            Instancia ya cargada con el YAML de configuración.

        Devuelve
        --------
        CaptureConfig
            Instancia inmutable lista para pasarse a ``VideoSource``.
        """
        source: int | str = cfg.get("capture.source", default=0)
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        return cls(
            source=source,
            width=cfg.get("capture.width", default=1280),
            height=cfg.get("capture.height", default=720),
            fps=cfg.get("capture.fps", default=30),
            window_title=cfg.get("capture.window_title", default="Motion Sentinel"),
        )

    # ------------------------------------------------------------------
    # Representación legible (útil en logs de arranque)
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"CaptureConfig("
            f"source={redact_url_credentials(self.source)!r}, "
            f"{self.width}x{self.height}@{self.fps}fps, "
            f"window={self.window_title!r})"
        )
