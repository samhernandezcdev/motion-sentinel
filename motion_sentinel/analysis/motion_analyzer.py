"""
Transforma las regiones crudas de movimiento (MotionRegion) en métricas
analíticas estructuradas (MotionMetrics).

Esta capa es puramente matemática: no toca OpenCV display, no emite logs
por frame, no modifica el estado del detector ni del renderer.
El único estado que mantiene el analizador es el historial de ratios de
movimiento, usado internamente para calcular la línea de base dinámica
del anomaly score.

Pipeline de llamada esperado
----------------------------
    mask, regions = detector.detect(processed_frame)
    metrics = analyzer.analyze(
        frame_shape=processed_frame.shape[:2],  # (height, width)
        regions=regions,
        mask=mask,          # opcional; reservado para cálculos futuros
    )

Evolución hacia fases futuras
------------------------------
    • Fase 2 - tracking  : MotionRegion ganará un campo `track_id` asignado
      por ObjectTracker (Kalman / SORT). MotionAnalyzer añadirá
      `velocity_vectors` y `centroid_displacements` a MotionMetrics.

    • Fase 3 - ML        : AnomalyDetector (analysis/anomaly.py) recibirá
      `metrics` como entrada y los completará con un campo `ml_anomaly_score`
      usando un autoencoder o LSTM entrenado sobre el historial de StatEngine.

    • Fase 4 - streaming : StatEngine (analysis/stat_engine.py) consumirá
      MotionMetrics vía EventBus para persistencia y agregación temporal.

Sin dependencias nuevas; solo stdlib + numpy.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from motion_sentinel.core.motion_detector import MotionRegion


@dataclass(frozen=True)
class MotionMetrics:
    """
    Métricas analíticas calculadas sobre un único frame.

    Todos los campos son de solo lectura (frozen dataclass) para que la
    instancia pueda pasarse entre módulos sin riesgo de mutación accidental.

    Atributos
    ----------
    active_regions : int
        Número de regiones de movimiento que superaron el umbral de área mínima.
    total_motion_area : float
        Suma de las áreas individuales de todas las regiones activas, en px².
    largest_region_area : float
        Área de la región más grande. 0.0 si no hay regiones.
    motion_ratio : float
        Proporción del frame cubierta por movimiento: A_total / A_frame.
        Rango [0.0, 1.0].
    average_region_area : float
        Media aritmética de las áreas de región. 0.0 si no hay regiones.
    anomaly_score : float
        Score compuesto normalizado en [0.0, 1.0].
        Combina motion_ratio (60 %) y contribución de la región más grande
        relativa al frame (40 %). Un score cercano a 1.0 indica un frame
        con alta cobertura o con una región dominante muy grande.
        Diseñado para ser reemplazado por un modelo TF en Fase 3.
    motion_detected : bool
        Verdadero si existe al menos una región activa en este frame.
    """

    active_regions: int
    total_motion_area: float
    largest_region_area: float
    motion_ratio: float
    average_region_area: float
    anomaly_score: float
    motion_detected: bool

    def __str__(self) -> str:
        return (
            f"MotionMetrics("
            f"regions={self.active_regions}, "
            f"ratio={self.motion_ratio:.4f}, "
            f"anomaly={self.anomaly_score:.4f}, "
            f"detected={self.motion_detected})"
        )


class MotionAnalyzer:
    """
    Calcula MotionMetrics a partir de una lista de MotionRegion y la forma del frame.

    Mantiene un historial deslizante de ``motion_ratio`` para estabilizar el
    anomaly_score mediante una línea de base adaptativa. El historial no
    persiste entre sesiones; se reinicia con :meth:`reset`.

    Parámetros
    ----------
    history_size : int
        Número de frames recientes guardados para la línea de base dinámica.
        Valor sugerido: 30-120 frames (1-4 segundos a 30 fps).
    anomaly_weight_ratio : float
        Peso del motion_ratio en el anomaly_score compuesto. El peso de la
        región más grande es ``1 - anomaly_weight_ratio``.
        Por defecto 0.6 (60 % ratio, 40 % región dominante).
    """

    def __init__(
        self,
        history_size: int = 30,
        anomaly_weight_ratio: float = 0.6,
    ) -> None:
        if not 0.0 < anomaly_weight_ratio < 1.0:
            msg = "anomaly_weight_ratio debe estar en el intervalo abierto (0, 1)."
            raise ValueError(msg)

        self._history_size = history_size
        self._w_ratio = anomaly_weight_ratio
        self._w_largest = 1.0 - anomaly_weight_ratio

        # Historial deslizante de motion_ratio por frame
        self._ratio_history: deque[float] = deque(maxlen=history_size)


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyze(
        self,
        frame_shape: tuple[int, int],
        regions: Sequence[MotionRegion],
        mask: np.ndarray | None = None,     # noqa: ARG002
    ) -> MotionMetrics:
        """
        Calcula y devuelve las métricas para el frame actual.

        Parámetros
        ----------
        frame_shape : tuple[int, int]
            Dimensiones (height, width) del frame procesado, en píxeles.
            Se obtiene directamente de ``processed_frame.shape[:2]``.
        regions : Sequence[MotionRegion]
            Regiones devueltas por ``FrameDifferenceMotionDetector.detect()``.
            Puede ser una lista vacía si no hay movimiento.
        mask : np.ndarray | None
            Máscara binaria opcional. No se usa en Fase 1; se acepta para
            mantener la firma estable durante la integración de fases futuras.

        Devuelve
        --------
        MotionMetrics
            Instancia inmutable con todas las métricas calculadas.
        """
        frame_area = _frame_area(frame_shape)

        total_area = _total_motion_area(regions)
        largest_area = _largest_region_area(regions)
        motion_ratio = _motion_ratio(total_area, frame_area)
        avg_area = _average_region_area(total_area, len(regions))

        self._ratio_history.append(motion_ratio)

        anomaly = _anomaly_score(
            motion_ratio=motion_ratio,
            largest_area=largest_area,
            frame_area=frame_area,
            w_ratio=self._w_ratio,
            w_largest=self._w_largest,
        )

        return MotionMetrics(
            active_regions=len(regions),
            total_motion_area=total_area,
            largest_region_area=largest_area,
            motion_ratio=motion_ratio,
            average_region_area=avg_area,
            anomaly_score=anomaly,
            motion_detected=len(regions) > 0,
        )


    @property
    def baseline_ratio(self) -> float:
        """
        Media del motion_ratio sobre los últimos ``history_size`` frames.

        Útil para comparar el frame actual con la actividad reciente de la
        escena. Devuelve 0.0 si el historial está vacío.

        Nota: Este valor es el punto de partida previsto para el reemplazo
        por un modelo de baseline estadístico en Fase 2 (StatEngine).
        """
        if not self._ratio_history:
            return 0.0
        return sum(self._ratio_history) / len(self._ratio_history)


    def reset(self) -> None:
        """
        Limpia el historial de ratios.

        Llamar tras cambiar de fuente de vídeo o al reanudar una sesión
        pausada, para evitar que la línea de base mezcle escenas distintas.
        """
        self._ratio_history.clear()


# ---------------------------------------------------------------------------
# Funciones de cálculo puras (sin estado, testeables de forma aislada)
# ---------------------------------------------------------------------------

def _frame_area(frame_shape: tuple[int, int]) -> float:
    """
    Área total del frame en px².

    Parámetros
    ----------
    frame_shape : tuple[int, int]
        (height, width) en píxeles.

    Devuelve
    --------
    float
        height * width. Siempre ≥ 1.0 para evitar división por cero en callers.
    """
    h, w = frame_shape
    return max(float(h * w), 1.0)


def _total_motion_area(regions: Sequence[MotionRegion]) -> float:
    """
    Σ region.area para todas las regiones activas.

    Devuelve 0.0 para una lista vacía, sin lanzar excepciones.
    """
    return sum(r.area for r in regions)


def _largest_region_area(regions: Sequence[MotionRegion]) -> float:
    """
    Área de la región más grande. Devuelve 0.0 si la lista está vacía.

    Las regiones ya llegan ordenadas de mayor a menor desde el detector,
    por lo que ``regions[0].area`` es suficiente; usamos ``max`` como
    contrato explícito que no asume el orden externo.
    """
    if not regions:
        return 0.0
    return max(r.area for r in regions)


def _motion_ratio(total_area: float, frame_area: float) -> float:
    """
    Proporción del frame cubierta por movimiento.

    R = A_total / A_frame, acotado a [0.0, 1.0].
    ``frame_area`` nunca llega a 0 gracias a :func:`_frame_area`.
    """
    ratio = total_area / frame_area
    return _clamp(ratio, 0.0, 1.0)


def _average_region_area(total_area: float, n_regions: int) -> float:
    """
    Media aritmética de las áreas de región.

    Devuelve 0.0 si ``n_regions`` es 0 para evitar ZeroDivisionError.
    """
    if n_regions == 0:
        return 0.0
    return total_area / n_regions


def _anomaly_score(
    motion_ratio: float,
    largest_area: float,
    frame_area: float,
    w_ratio: float,
    w_largest: float,
) -> float:
    """
    Score de anomalía compuesto, normalizado en [0.0, 1.0].

    Fórmula
    -------
    ::

        largest_ratio = largest_area / frame_area
        score = w_ratio * motion_ratio + w_largest * largest_ratio

    Ambos sumandos ya están en [0, 1], y los pesos suman 1.0, por lo que
    el resultado es naturalmente acotado. El ``_clamp`` final protege contra
    errores de punto flotante en los bordes.

    Diseño para fases futuras
    -------------------------
    En Fase 3 este valor se reemplazará por la salida de un autoencoder
    (reconstruction_error normalizado) entrenado sobre el historial de
    StatEngine. La firma de ``MotionMetrics.anomaly_score`` no cambia.
    """
    largest_ratio = _clamp(largest_area / frame_area, 0.0, 1.0)
    raw = w_ratio * motion_ratio + w_largest * largest_ratio
    return _clamp(raw, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    """Acota ``value`` al intervalo cerrado [low, high]."""
    return max(low, min(high, value))
