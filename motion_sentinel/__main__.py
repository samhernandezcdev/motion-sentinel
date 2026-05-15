import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from motion_sentinel.analysis.alert_manager import AlertEvent, AlertManager
from motion_sentinel.analysis.motion_analyzer import MotionAnalyzer
from motion_sentinel.analysis.stat_engine import StatEngine
from motion_sentinel.capture.config import CaptureConfig
from motion_sentinel.capture.frame_buffer import BufferedVideoSource, FrameBufferConfig
from motion_sentinel.capture.video_source import VideoSource, VideoSourceError
from motion_sentinel.common.config_manager import ConfigManager
from motion_sentinel.common.logger import get_logger, setup_logging
from motion_sentinel.core.motion_detector import FrameDifferenceMotionDetector
from motion_sentinel.core.preprocessor import preprocess_for_motion
from motion_sentinel.core.roi_manager import ROI, ROIHit, ROIManager
from motion_sentinel.output.recorder import RecordingConfig, SnapshotRecorder
from motion_sentinel.output.renderer import RendererConfig, render_motion_overlay

_CONFIG_PATH = Path("config/default.yaml")


def main() -> None:
    # 1. Cargar configuración
    cfg = _load_app_config()

    # 2. Inicializar logging
    log = _setup_app_logging(cfg)

    capture_config = CaptureConfig.from_config(cfg)

    log.info(
        "Inicializando Motion Sentinel",
        version=cfg.get("app.version"),
        log_level=cfg.get("app.log_level", default="INFO"),
    )

    # 3. Construir pipeline
    detector = _build_detector(cfg)
    analyzer = _build_analyzer(cfg)
    alert_manager = _build_alert_manager(cfg)
    renderer_config = _build_renderer_config(cfg)
    recorder = _build_recorder(cfg)
    roi_manager = _build_roi_manager(cfg, log)
    stats = StatEngine()

    resize_width: int | None = cfg.get("detection.resize_width", default=640)
    blur_kernel: int = cfg.get("detection.blur_kernel", default=21)

    # 4. Seleccionar fuente de frames según configuración
    use_buffer: bool = cfg.get("frame_buffer.enabled", default=False)

    log.info("Modo de captura seleccionado", buffered=use_buffer)

    # 5. Abrir cámara y mostrar vídeo en tiempo real
    try:
        frame_iter = _open_frame_source(cfg, capture_config, use_buffer)
        log.info("Presiona 'q' para salir")

        for frame in frame_iter:
            # Detección
            processed = preprocess_for_motion(
                frame, resize_width=resize_width, blur_kernel=blur_kernel,
            )
            mask, regions = detector.detect(processed)

            # Observación por ROI (no modifica regions ni mask)
            roi_hits = roi_manager.analyze_regions(regions, frame_shape=processed.shape[:2])

            _log_roi_activity(log, roi_hits)

            # Análisis
            metrics = analyzer.analyze(
                frame_shape=processed.shape[:2], regions=regions, mask=mask,
            )

            # Renderizado
            display_frame = _prepare_display_frame(frame, processed, resize_width)
            rendered = render_motion_overlay(
                display_frame, regions,
                mask=mask, metrics=metrics, config=renderer_config,
                rois=roi_manager.enabled_rois(),
                roi_hits=roi_hits,
            )

            # Alertas y snapshots
            alert = alert_manager.evaluate(metrics)
            snapshot_saved = False

            if alert is not None:
                _log_alert(log, alert)
                saved_path = recorder.save_snapshot(rendered, alert)
                snapshot_saved = saved_path is not None

            stats.update(metrics=metrics, alert=alert, snapshot_saved=snapshot_saved)

            cv2.imshow(capture_config.window_title, rendered)

            # Salir al pulsar 'q' (waitKey devuelve -1 si no hay tecla)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Se ha pulsado la tecla de salida, cerrando sesión")
                break

    except VideoSourceError as exc:
        log.error("No se ha podido abrir la fuente de vídeo", error=str(exc))
        sys.exit(1)

    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario (Ctrl+C)")

    finally:
        try:
            cv2.destroyAllWindows()
        except Exception as exc:
            log.warning("No se pudieron cerrar ventanas OpenCV", error=str(exc))

        _log_session_summary(log, stats)
        log.info("Motion Sentinel detenido")


# ---------------------------------------------------------------------------
# Fuente de frames: síncrona o buffered
# ---------------------------------------------------------------------------

def _open_frame_source(
    cfg: ConfigManager,
    capture_config: CaptureConfig,
    use_buffer: bool,
) -> Iterator[np.ndarray]:
    """
    Devuelve un iterador de frames según el modo configurado.

    - ``use_buffer=False``: VideoSource clásico, single-thread.
    - ``use_buffer=True``:  BufferedVideoSource con hilo productor dedicado.

    El cierre de recursos se gestiona aquí con ``try/finally`` para que
    el loop principal sea agnóstico al tipo de fuente.
    """
    video_source = VideoSource(
        source=capture_config.source,
        width=capture_config.width,
        height=capture_config.height,
        fps=capture_config.fps,
    )

    if not use_buffer:
        # Modo síncrono: comportamiento original, sin cambios
        with video_source:
            yield from video_source.frames()
        return

    # Modo buffered: hilo productor + queue thread-safe
    buffer_config = FrameBufferConfig(
        maxsize=cfg.get("frame_buffer.maxsize", default=8),
        drop_oldest=cfg.get("frame_buffer.drop_oldest", default=True),
    )
    buffered = BufferedVideoSource(video_source, buffer_config)

    buffered.start()
    try:
        yield from buffered.frames()
    finally:
        buffered.stop()


# ---------------------------------------------------------------------------
# Configuración y logging
# ---------------------------------------------------------------------------


def _load_app_config() -> ConfigManager:
    """Carga el YAML de configuración desde la ruta por defecto."""
    return ConfigManager(_CONFIG_PATH)


def _setup_app_logging(cfg: ConfigManager) -> Any:
    """Inicializa structlog y devuelve el logger del módulo principal."""
    log_level: str = cfg.get("app.log_level", default="INFO")
    setup_logging(level=log_level)
    return get_logger(__name__)


# ---------------------------------------------------------------------------
# Construcción del pipeline
# ---------------------------------------------------------------------------

def _build_detector(cfg: ConfigManager) -> FrameDifferenceMotionDetector:
    """Instancia el detector de movimiento con parámetros de config."""
    return FrameDifferenceMotionDetector(
        threshold=cfg.get("detection.threshold", default=25),
        min_area=cfg.get("detection.min_area", default=500),
        dilation_iterations=cfg.get("detection.dilation_iterations", default=2),
    )


def _build_analyzer(cfg: ConfigManager) -> MotionAnalyzer:
    """Instancia el analizador de métricas de movimiento."""
    return MotionAnalyzer(
        history_size=cfg.get("analyzer.history_size", default=30),
        anomaly_weight_ratio=cfg.get("analyzer.anomaly_weight_ratio", default=0.6),
    )


def _build_alert_manager(cfg: ConfigManager) -> AlertManager:
    """Instancia el gestor de alertas con umbrales y cooldown de config."""
    return AlertManager(
        low_threshold=cfg.get("alerts.low_threshold", default=0.05),
        medium_threshold=cfg.get("alerts.medium_threshold", default=0.15),
        high_threshold=cfg.get("alerts.high_threshold", default=0.30),
        anomaly_threshold=cfg.get("alerts.anomaly_threshold", default=0.50),
        cooldown_frames=cfg.get("alerts.cooldown_frames", default=15),
    )


def _build_renderer_config(cfg: ConfigManager) -> RendererConfig:
    """Construye la configuración visual del renderer."""
    return RendererConfig(
        show_mask_inset=cfg.get("renderer.show_mask_inset", default=False),
    )


def _build_recorder(cfg: ConfigManager) -> SnapshotRecorder:
    """Instancia el recorder de snapshots con directorio y calidad de config."""
    return SnapshotRecorder(
        RecordingConfig(
            output_dir=Path(cfg.get("recorder.output_dir", default="data/snapshots")),
            jpeg_quality=cfg.get("recorder.jpeg_quality", default=90),
            filename_prefix=cfg.get("recorder.filename_prefix", default="event"),
        )
    )


def _build_roi_manager(cfg: ConfigManager, log: Any) -> ROIManager:
    """
    Construye el ROIManager leyendo la lista ``rois`` del YAML.

    Cada entrada debe tener al menos: name, x, y, width, height.
    Los campos ``weight`` y ``enabled`` son opcionales (defaults: 1.0 / True).
    Si la clave ``rois`` no existe o está vacía, devuelve un manager sin ROIs
    (passthrough: ``analyze_regions`` devolverá ``[]`` en cada frame).
    """
    manager = ROIManager()
    roi_list: list[dict] = cfg.get("rois", default=[]) or []

    for entry in roi_list:
        try:
            manager.add_roi(ROI(**entry))
        except TypeError as exc:
            log.warning("ROI inválido en config, ignorado", entry=entry, error=str(exc))

    if manager.enabled_rois():
        log.info("ROIs de observación cargados", total=len(manager.enabled_rois()))
    else:
        log.debug("Sin ROIs configurados, observación de zonas desactivada")

    return manager


# ---------------------------------------------------------------------------
# Helpers del loop
# ---------------------------------------------------------------------------

def _log_roi_activity(log: Any, roi_hits: list[ROIHit]) -> None:
    """Emite un log por cada ROI con actividad en el frame actual."""
    for hit in roi_hits:
        log.info(
            "roi_activity",
            roi=hit.roi_name,
            regions=hit.region_count,
            area=int(hit.total_area),
            weighted_score=f"{hit.weighted_score:.1f}",
        )


def _prepare_display_frame(
    frame: np.ndarray,
    processed: np.ndarray,
    resize_width: int | None,
) -> np.ndarray:
    """
    Redimensiona el frame original a las dimensiones del frame procesado.

    Si ``resize_width`` es ``None`` el detector operó a resolución original
    y no hay nada que reescalar; se devuelve el frame sin copiar.
    """
    if resize_width is None:
        return frame
    return cv2.resize(
        frame,
        (processed.shape[1], processed.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


def _log_alert(log: Any, alert: AlertEvent) -> None:
    """Emite una línea de log estructurado para un AlertEvent."""
    log.info(
        "motion_alert",
        severity=alert.severity.name,
        message=alert.message,
        score=f"{alert.anomaly_score:.4f}",
        motion_ratio=f"{alert.motion_ratio:.4f}",
        regions=alert.active_regions,
        should_record=alert.should_record,
    )


def _log_session_summary(log: Any, stats: StatEngine) -> None:
    """Emite el resumen de sesión al cerrar el pipeline."""
    summary = stats.summary()
    log.info(
        "session_summary",
        frames=summary.total_frames,
        motion_frames=summary.motion_frames,
        motion_ratio=f"{summary.motion_frame_ratio:.4f}",
        alerts=summary.alert_count,
        snapshots=summary.snapshot_count,
        max_score=f"{summary.max_anomaly_score:.4f}",
        avg_score=f"{summary.avg_anomaly_score:.4f}",
    )


if __name__ == "__main__":
    main()
