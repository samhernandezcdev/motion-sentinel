"""Runtime pipeline orchestration for Motion Sentinel."""

from __future__ import annotations

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
from motion_sentinel.capture.video_source import VideoSource
from motion_sentinel.common.app_config import AppConfig
from motion_sentinel.common.logger import get_logger
from motion_sentinel.core.motion_detector import FrameDifferenceMotionDetector
from motion_sentinel.core.preprocessor import preprocess_for_motion
from motion_sentinel.core.roi_manager import ROI, ROIHit, ROIManager
from motion_sentinel.output.recorder import RecordingConfig, SnapshotRecorder
from motion_sentinel.output.renderer import RendererConfig, render_motion_overlay


class SessionRunner:
    """Owns one Motion Sentinel processing session."""

    def __init__(self, config: AppConfig, log: Any | None = None) -> None:
        self._config = config
        self._log = log or get_logger(__name__)

    def run(self) -> None:
        """Run the capture, detection, rendering, alert, and recording loop."""
        capture_config = self._build_capture_config()
        detector = self._build_detector()
        analyzer = self._build_analyzer()
        alert_manager = self._build_alert_manager()
        renderer_config = self._build_renderer_config()
        recorder = self._build_recorder()
        roi_manager = self._build_roi_manager()
        stats = StatEngine()

        self._log.info(
            "Modo de captura seleccionado",
            buffered=self._config.frame_buffer.enabled,
            headless=self._config.headless,
        )

        try:
            frame_iter = self._open_frame_source(capture_config)
            if not self._config.headless:
                self._log.info("Presiona 'q' para salir")

            for frame in frame_iter:
                source_shape = frame.shape[:2]

                processed = preprocess_for_motion(
                    frame,
                    resize_width=self._config.detection.resize_width,
                    blur_kernel=self._config.detection.blur_kernel,
                )
                processed_shape = processed.shape[:2]
                mask, regions = detector.detect(processed)

                roi_hits = roi_manager.analyze_regions(
                    regions,
                    frame_shape=processed_shape,
                    source_shape=source_shape,
                )
                self._log_roi_activity(roi_hits)

                metrics = analyzer.analyze(
                    frame_shape=processed_shape,
                    regions=regions,
                    mask=mask,
                )

                display_frame = self._prepare_display_frame(frame, processed)
                rendered = render_motion_overlay(
                    display_frame,
                    regions,
                    mask=mask,
                    metrics=metrics,
                    config=renderer_config,
                    rois=roi_manager.scaled_enabled_rois(source_shape, processed_shape),
                    roi_hits=roi_hits,
                )

                alert = alert_manager.evaluate(metrics)
                snapshot_saved = False

                if alert is not None:
                    self._log_alert(alert)
                    saved_path = recorder.save_snapshot(rendered, alert)
                    snapshot_saved = saved_path is not None

                stats.update(
                    metrics=metrics,
                    alert=alert,
                    snapshot_saved=snapshot_saved,
                )

                if self._show_frame(capture_config.window_title, rendered):
                    self._log.info("Se ha pulsado la tecla de salida, cerrando sesión")
                    break

        except KeyboardInterrupt:
            self._log.info("Interrumpido por el usuario (Ctrl+C)")

        finally:
            if not self._config.headless:
                try:
                    cv2.destroyAllWindows()
                except Exception as exc:
                    self._log.warning(
                        "No se pudieron cerrar ventanas OpenCV", error=str(exc)
                    )

            self._log_session_summary(stats)
            self._log.info("Motion Sentinel detenido")

    def _open_frame_source(self, capture_config: CaptureConfig) -> Iterator[np.ndarray]:
        video_source = VideoSource(
            source=capture_config.source,
            width=capture_config.width,
            height=capture_config.height,
            fps=capture_config.fps,
        )

        if not self._config.frame_buffer.enabled:
            with video_source:
                yield from video_source.frames()
            return

        buffer_config = FrameBufferConfig(
            maxsize=self._config.frame_buffer.maxsize,
            drop_oldest=self._config.frame_buffer.drop_oldest,
        )
        buffered = BufferedVideoSource(video_source, buffer_config)

        buffered.start()
        try:
            yield from buffered.frames()
        finally:
            buffered.stop()

    def _show_frame(self, window_title: str, rendered: np.ndarray) -> bool:
        """Display one frame. Returns True when the user requested exit."""
        if self._config.headless:
            return False

        cv2.imshow(window_title, rendered)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def _prepare_display_frame(
        self,
        frame: np.ndarray,
        processed: np.ndarray,
    ) -> np.ndarray:
        if self._config.detection.resize_width is None:
            return frame
        return cv2.resize(
            frame,
            (processed.shape[1], processed.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    def _build_capture_config(self) -> CaptureConfig:
        capture = self._config.capture
        return CaptureConfig(
            source=capture.source,
            width=capture.width,
            height=capture.height,
            fps=capture.fps,
            window_title=capture.window_title,
        )

    def _build_detector(self) -> FrameDifferenceMotionDetector:
        detection = self._config.detection
        return FrameDifferenceMotionDetector(
            threshold=detection.threshold,
            min_area=detection.min_area,
            dilation_iterations=detection.dilation_iterations,
        )

    def _build_analyzer(self) -> MotionAnalyzer:
        analyzer = self._config.analyzer
        return MotionAnalyzer(
            history_size=analyzer.history_size,
            anomaly_weight_ratio=analyzer.anomaly_weight_ratio,
        )

    def _build_alert_manager(self) -> AlertManager:
        alerts = self._config.alerts
        return AlertManager(
            low_threshold=alerts.low_threshold,
            medium_threshold=alerts.medium_threshold,
            high_threshold=alerts.high_threshold,
            anomaly_threshold=alerts.anomaly_threshold,
            cooldown_frames=alerts.cooldown_frames,
        )

    def _build_renderer_config(self) -> RendererConfig:
        renderer = self._config.renderer
        return RendererConfig(
            show_mask_inset=renderer.show_mask_inset,
            show_rois=renderer.show_rois,
        )

    def _build_recorder(self) -> SnapshotRecorder:
        recorder = self._config.recorder
        return SnapshotRecorder(
            RecordingConfig(
                output_dir=Path(recorder.output_dir),
                image_format=recorder.image_format,
                jpeg_quality=recorder.jpeg_quality,
                filename_prefix=recorder.filename_prefix,
            )
        )

    def _build_roi_manager(self) -> ROIManager:
        manager = ROIManager()
        for entry in self._config.rois:
            manager.add_roi(
                ROI(
                    name=entry.name,
                    x=entry.x,
                    y=entry.y,
                    width=entry.width,
                    height=entry.height,
                    weight=entry.weight,
                    enabled=entry.enabled,
                )
            )

        if manager.enabled_rois():
            self._log.info(
                "ROIs de observación cargados", total=len(manager.enabled_rois())
            )
        else:
            self._log.debug("Sin ROIs configurados, observación de zonas desactivada")

        return manager

    def _log_roi_activity(self, roi_hits: list[ROIHit]) -> None:
        for hit in roi_hits:
            self._log.info(
                "roi_activity",
                roi=hit.roi_name,
                regions=hit.region_count,
                area=int(hit.total_area),
                weighted_score=f"{hit.weighted_score:.1f}",
            )

    def _log_alert(self, alert: AlertEvent) -> None:
        self._log.info(
            "motion_alert",
            severity=alert.severity.name,
            message=alert.message,
            score=f"{alert.anomaly_score:.4f}",
            motion_ratio=f"{alert.motion_ratio:.4f}",
            regions=alert.active_regions,
            should_record=alert.should_record,
        )

    def _log_session_summary(self, stats: StatEngine) -> None:
        summary = stats.summary()
        self._log.info(
            "session_summary",
            frames=summary.total_frames,
            motion_frames=summary.motion_frames,
            motion_ratio=f"{summary.motion_frame_ratio:.4f}",
            alerts=summary.alert_count,
            snapshots=summary.snapshot_count,
            max_score=f"{summary.max_anomaly_score:.4f}",
            avg_score=f"{summary.avg_anomaly_score:.4f}",
        )
