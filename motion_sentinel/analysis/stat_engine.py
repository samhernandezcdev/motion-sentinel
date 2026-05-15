"""
Acumula estadísticas de sesión en memoria a partir de ``MotionMetrics``
y ``AlertEvent``.

Responsabilidades
-----------------
- Contar frames totales, frames con movimiento, alertas y snapshots.
- Calcular promedios de ``motion_ratio`` y ``anomaly_score`` de forma
  incremental (sin guardar el historial completo de frames).
- Registrar el ``anomaly_score`` máximo observado en la sesión.
- Exponer un resumen inmutable (``SessionStats``) en cualquier momento.

Lo que NO hace (fases futuras)
-------------------------------
- Persistencia en SQLite / PostgreSQL  → Fase 2: ``StorageAdapter``.
- Emisión al EventBus                  → Fase 2: ``StatEngine`` como subscriber.
- Exportación a CSV / JSON             → Fase 2: método ``export(path)``.
- Estadísticas por ROI o por track_id  → Fase 3: tras integrar ``ObjectTracker``.
- Series temporales para ML            → Fase 3: historial de ratios como
                                         input de ``AnomalyDetector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from motion_sentinel.analysis.alert_manager import AlertEvent
    from motion_sentinel.analysis.motion_analyzer import MotionMetrics


@dataclass(frozen=True)
class SessionStats:
    """
    Resumen inmutable de la sesión actual.

    Producido por ``StatEngine.summary()`` y seguro para pasar a cualquier
    módulo sin riesgo de mutación.

    Atributos
    ----------
    total_frames : int
        Frames procesados desde la última llamada a ``reset()``.
    motion_frames : int
        Frames en los que ``MotionMetrics.motion_detected`` fue ``True``.
    alert_count : int
        Número de ``AlertEvent`` no nulos recibidos por ``update()``.
    snapshot_count : int
        Número de veces que ``update()`` recibió ``snapshot_saved=True``.
    max_anomaly_score : float
        Mayor ``anomaly_score`` observado en la sesión. ``0.0`` si no hubo
        ningún frame aún.
    avg_motion_ratio : float
        Media aritmética de ``motion_ratio`` sobre todos los frames.
        ``0.0`` si no hubo ningún frame.
    avg_anomaly_score : float
        Media aritmética de ``anomaly_score`` sobre todos los frames.
        ``0.0`` si no hubo ningún frame.
    """

    total_frames: int
    motion_frames: int
    alert_count: int
    snapshot_count: int
    max_anomaly_score: float
    avg_motion_ratio: float
    avg_anomaly_score: float

    # ------------------------------------------------------------------
    # Métricas derivadas (propiedades calculadas, sin estado extra)
    # ------------------------------------------------------------------

    @property
    def motion_frame_ratio(self) -> float:
        """Proporción de frames con movimiento sobre el total. ``0.0`` si no hay frames."""
        if self.total_frames == 0:
            return 0.0
        return self.motion_frames / self.total_frames

    @property
    def alert_rate(self) -> float:
        """Alertas por frame. Útil para comparar sesiones de distinta duración."""
        if self.total_frames == 0:
            return 0.0
        return self.alert_count / self.total_frames

    def __str__(self) -> str:
        return (
            f"SessionStats("
            f"frames={self.total_frames}, "
            f"motion={self.motion_frames} ({self.motion_frame_ratio:.1%}), "
            f"alerts={self.alert_count}, "
            f"snapshots={self.snapshot_count}, "
            f"max_score={self.max_anomaly_score:.4f}, "
            f"avg_score={self.avg_anomaly_score:.4f})"
        )


# ---------------------------------------------------------------------------
# StatEngine
# ---------------------------------------------------------------------------


class StatEngine:
    """
    Acumula estadísticas de sesión frame a frame.

    Usa medias incrementales (Welford simplificado: suma + contador) para
    mantener ``O(1)`` en memoria independientemente de la duración de la
    sesión. No guarda el historial de valores individuales.

    No tiene dependencias de OpenCV, threading ni I/O: es un acumulador
    de números puro, testeable con cualquier combinación de métricas sintéticas.
    """

    def __init__(self) -> None:
        self._reset_state()

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def update(
        self,
        metrics: MotionMetrics,
        alert: AlertEvent | None = None,
        snapshot_saved: bool = False,
    ) -> None:
        """
        Registra las métricas del frame actual en el estado de sesión.

        Debe llamarse una vez por frame, tras ``MotionAnalyzer.analyze()``
        y (opcionalmente) tras ``AlertManager.evaluate()`` y
        ``SnapshotRecorder.save_snapshot()``.

        Parámetros
        ----------
        metrics : MotionMetrics
            Métricas del frame producidas por ``MotionAnalyzer``.
        alert : AlertEvent | None
            Evento de alerta del frame, o ``None`` si no hubo alerta.
        snapshot_saved : bool
            ``True`` si ``SnapshotRecorder`` guardó un archivo en este frame.
        """
        self._total_frames += 1

        if metrics.motion_detected:
            self._motion_frames += 1

        # Sumas acumuladas para cálculo de medias en O(1) sin historial
        self._sum_motion_ratio += metrics.motion_ratio
        self._sum_anomaly_score += metrics.anomaly_score

        if metrics.anomaly_score > self._max_anomaly_score:
            self._max_anomaly_score = metrics.anomaly_score

        if alert is not None:
            self._alert_count += 1

        if snapshot_saved:
            self._snapshot_count += 1

    def summary(self) -> SessionStats:
        """
        Devuelve un resumen inmutable del estado actual de la sesión.

        Puede llamarse en cualquier momento sin alterar el estado interno.
        Seguro para llamar desde el renderer, un logger periódico o, en
        Fase 2, desde un endpoint de ``DashboardServer``.
        """
        n = self._total_frames
        return SessionStats(
            total_frames=n,
            motion_frames=self._motion_frames,
            alert_count=self._alert_count,
            snapshot_count=self._snapshot_count,
            max_anomaly_score=self._max_anomaly_score,
            avg_motion_ratio=self._sum_motion_ratio / n if n > 0 else 0.0,
            avg_anomaly_score=self._sum_anomaly_score / n if n > 0 else 0.0,
        )

    def reset(self) -> None:
        """
        Reinicia todos los contadores y acumuladores a cero.

        Útil al cambiar de fuente de vídeo o al iniciar una nueva sesión
        de grabación sin recrear la instancia.
        """
        self._reset_state()

    # ------------------------------------------------------------------
    # Estado interno
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Inicializa (o reinicia) todos los campos de estado."""
        self._total_frames: int = 0
        self._motion_frames: int = 0
        self._alert_count: int = 0
        self._snapshot_count: int = 0
        self._max_anomaly_score: float = 0.0
        self._sum_motion_ratio: float = 0.0
        self._sum_anomaly_score: float = 0.0
