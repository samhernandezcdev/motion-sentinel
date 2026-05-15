"""
Guarda snapshots JPEG en disco cuando una alerta lo requiere.

Responsabilidades
-----------------
- Decidir si un ``AlertEvent`` justifica escritura a disco (``should_record``).
- Construir nombres de archivo deterministas con timestamp UTC y severidad.
- Crear el directorio de salida si no existe.
- Escribir el frame con ``cv2.imwrite`` y calidad JPEG configurable.
- Absorber errores de I/O: un fallo de escritura no debe derribar el pipeline.

Lo que NO hace (fases futuras)
-------------------------------
- Clips MP4 / video                → Fase 2: método ``start_clip()`` / ``stop_clip()``
                                     usando ``cv2.VideoWriter``.
- Retención automática (purge)     → Fase 2: ``StorageAdapter`` + política de días.
- Emisión de eventos al EventBus   → Fase 2: el recorder se suscribe a
                                     ``AlertSeverity.HIGH`` y ``ANOMALY`` vía bus.
- Subida a almacenamiento remoto   → Fase 3: S3 / GCS adapter sobre ``StorageAdapter``.
- Threading / cola de escritura    → Fase 2: ``queue.Queue`` + worker thread para
                                     no bloquear el loop de captura en discos lentos.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from motion_sentinel.common.logger import get_logger

if TYPE_CHECKING:
    from motion_sentinel.analysis.alert_manager import AlertEvent

_log = get_logger(__name__)


@dataclass
class RecordingConfig:
    """
    Parámetros de escritura para ``SnapshotRecorder``.

    Atributos
    ----------
    output_dir : Path
        Directorio raíz donde se guardan los snapshots.
        Se crea automáticamente si no existe.
    image_format : str
        Extensión del archivo de imagen. Solo ``"jpg"`` está validado;
        ``"png"`` funciona pero ignora ``jpeg_quality``.
    jpeg_quality : int
        Calidad JPEG en el rango [0, 100]. 90 ofrece buen equilibrio entre
        tamaño y fidelidad para análisis posterior. A 100 los archivos son
        ~3x más grandes sin ganancia perceptual para visión computacional.
    filename_prefix : str
        Prefijo que encabeza todos los nombres de archivo generados.
    """

    output_dir: Path
    image_format: str = "jpg"
    jpeg_quality: int = 90
    filename_prefix: str = "event"

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if not 0 <= self.jpeg_quality <= 100:
            msg = f"jpeg_quality={self.jpeg_quality} debe estar en [0, 100]."
            raise ValueError(msg)
        if not self.filename_prefix:
            msg = "filename_prefix no puede ser vacío."
            raise ValueError(msg)


class SnapshotRecorder:
    """
    Escribe snapshots JPEG a disco cuando ``alert.should_record`` es ``True``.

    El recorder es stateless entre llamadas: no mantiene buffers de frames
    ni hilos. Cada llamada a ``save_snapshot`` es atómica y síncrona.

    Parámetros
    ----------
    config : RecordingConfig
        Configuración de directorio, formato y calidad.

    Ejemplo de uso
    --------------
    ::

        config = RecordingConfig(output_dir=Path("data/snapshots"))
        recorder = SnapshotRecorder(config)

        alert = alert_manager.evaluate(metrics)
        if alert is not None:
            saved_path = recorder.save_snapshot(frame, alert)
    """

    def __init__(self, config: RecordingConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        frame: np.ndarray,
        alert: AlertEvent,
    ) -> Path | None:
        """
        Guarda el frame como imagen si ``alert.should_record`` es ``True``.

        Parámetros
        ----------
        frame : np.ndarray
            Frame BGR (H, W, 3) uint8 tal como llega del renderer o del
            pipeline de captura. No se modifica.
        alert : AlertEvent
            Evento producido por ``AlertManager.evaluate()``.

        Devuelve
        --------
        Path
            Ruta absoluta del archivo guardado.
        None
            Si ``alert.should_record`` es ``False`` o si ocurre un error
            de escritura (el error se registra en el log pero no se propaga).
        """
        if not alert.should_record:
            return None

        self._ensure_output_dir()

        filepath = self._config.output_dir / self._build_filename(alert)

        try:
            encode_params = self._encode_params()
            success = cv2.imwrite(str(filepath), frame, encode_params)

            if not success:
                _log.warning(
                    "cv2.imwrite devolvió False — snapshot no guardado",
                    path=str(filepath),
                    severity=alert.severity.name,
                )
                return None

        except OSError as exc:
            _log.error(
                "Error de I/O al guardar snapshot",
                path=str(filepath),
                severity=alert.severity.name,
                error=str(exc),
            )
            return None

        _log.info(
            "Snapshot guardado",
            path=str(filepath),
            severity=alert.severity.name,
            score=f"{alert.anomaly_score:.4f}",
            regions=alert.active_regions,
        )

        return filepath

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _build_filename(self, alert: AlertEvent) -> str:
        """
        Construye el nombre de archivo con timestamp UTC y severidad.

        Formato
        -------
        ``{prefix}_{YYYYMMDD}_{HHMMSS}_{mmm}_{SEVERITY}.{ext}``

        Ejemplo
        -------
        ``event_20260515_221530_123_HIGH.jpg``

        El timestamp es UTC para que los archivos sean ordenables y
        comparables entre zonas horarias sin ambigüedad. Los milisegundos
        evitan colisiones cuando dos alertas se emiten dentro del mismo segundo
        (posible a 30 fps con cooldown_frames=1).
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        millis = f"{now.microsecond // 1000:03d}"
        severity = alert.severity.name

        return (
            f"{self._config.filename_prefix}"
            f"_{timestamp}"
            f"_{millis}"
            f"_{severity}"
            f".{self._config.image_format}"
        )

    def _ensure_output_dir(self) -> None:
        """
        Crea ``output_dir`` y todos sus padres si no existen.

        Usa ``exist_ok=True`` para que llamadas concurrentes (Fase 2) no
        lancen ``FileExistsError`` en una condición de carrera.
        """
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def _encode_params(self) -> list[int]:
        """
        Construye la lista de parámetros de codificación para ``cv2.imwrite``.

        Para JPEG devuelve ``[cv2.IMWRITE_JPEG_QUALITY, quality]``.
        Para otros formatos devuelve lista vacía (defaults de OpenCV).
        """
        if self._config.image_format.lower() in {"jpg", "jpeg"}:
            return [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality]
        return []
