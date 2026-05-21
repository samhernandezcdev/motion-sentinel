"""
Convierte un ``MotionMetrics`` en un ``AlertEvent`` estructurado.

Responsabilidades
-----------------
- Clasificar la severidad de la actividad según ``anomaly_score``.
- Suprimir alertas repetitivas mediante un cooldown por frames.
- Indicar qué severidades justifican grabación (``should_record``).
- Mantenerse desacoplado: no sabe nada de cámaras, renderers ni EventBus.

Lo que NO hace (fases futuras)
-------------------------------
- No emite eventos al EventBus  → Fase 2 (event_bus.py).
- No activa RecordingManager    → Fase 2 (recorder.py).
- No hace inferencia ML         → Fase 3 (anomaly.py + TensorFlow).
- No envía notificaciones       → Fase 2 (notif.py).
"""

from dataclasses import dataclass
from enum import Enum, auto

from motion_sentinel.analysis.motion_analyzer import MotionMetrics


class AlertSeverity(Enum):
    """
    Niveles de severidad de una alerta de movimiento.

    El orden numérico implícito no se usa intencionalmente; la comparación
    entre niveles se hace siempre por identidad (``severity is AlertSeverity.HIGH``),
    no por valor, para evitar acoplamientos frágiles con los enteros de ``auto()``.
    """

    NONE = auto()  # Sin movimiento - no genera AlertEvent
    LOW = auto()  # Actividad mínima por encima del umbral de detección
    MEDIUM = auto()  # Actividad moderada, monitorizar
    HIGH = auto()  # Actividad elevada, candidata a grabación
    ANOMALY = auto()  # Patrón fuera de lo esperado, grabación recomendada


@dataclass(frozen=True)
class AlertEvent:
    """
    Evento de alerta inmutable producido por ``AlertManager.evaluate()``.

    Diseñado para viajar por el sistema sin ser mutado: puede pasarse al
    renderer, al logger o —en Fase 2— al EventBus sin riesgo de modificación
    accidental.

    Atributos
    ----------
    severity : AlertSeverity
        Nivel de severidad clasificado por ``AlertManager``.
    message : str
        Descripción técnica legible del evento.
    anomaly_score : float
        Valor de ``MotionMetrics.anomaly_score`` en el momento de la alerta.
        Rango [0.0, 1.0].
    motion_ratio : float
        Proporción del frame cubierta por movimiento en el momento de la alerta.
        Rango [0.0, 1.0].
    active_regions : int
        Número de regiones activas en el frame que disparó la alerta.
    should_record : bool
        ``True`` si la severidad justifica iniciar grabación (HIGH o ANOMALY).
        El ``RecordingManager`` de Fase 2 leerá este campo para decidir.
    """

    severity: AlertSeverity
    message: str
    anomaly_score: float
    motion_ratio: float
    active_regions: int
    should_record: bool

    def __str__(self) -> str:
        return (
            f"AlertEvent("
            f"severity={self.severity.name}, "
            f"score={self.anomaly_score:.4f}, "
            f"ratio={self.motion_ratio:.4f}, "
            f"regions={self.active_regions}, "
            f"record={self.should_record})"
        )


class AlertManager:
    """
    Evalúa ``MotionMetrics`` frame a frame y emite ``AlertEvent`` cuando procede.

    Cooldown
    --------
    Una vez emitida una alerta, el manager entra en ``cooldown_frames`` frames
    de silencio. Esto evita que un evento prolongado (p. ej. una persona
    cruzando el campo visual durante 2 segundos) genere decenas de alertas
    idénticas. El cooldown se decrementa en cada llamada a ``evaluate()``,
    independientemente de si hay movimiento o no.

    Parámetros
    ----------
    low_threshold : float
        ``anomaly_score`` mínimo para emitir ``LOW``. Por debajo → silencio.
    medium_threshold : float
        ``anomaly_score`` mínimo para emitir ``MEDIUM``.
    high_threshold : float
        ``anomaly_score`` mínimo para emitir ``HIGH``.
    anomaly_threshold : float
        ``anomaly_score`` mínimo para emitir ``ANOMALY``.
    cooldown_frames : int
        Número de frames de silencio tras emitir una alerta.
        A 30 fps, el valor por defecto (15) equivale a ~0.5 segundos.
    """

    def __init__(
        self,
        low_threshold: float = 0.05,
        medium_threshold: float = 0.15,
        high_threshold: float = 0.30,
        anomaly_threshold: float = 0.50,
        cooldown_frames: int = 15,
    ) -> None:
        _validate_thresholds(
            low_threshold,
            medium_threshold,
            high_threshold,
            anomaly_threshold,
        )

        self._low_threshold = low_threshold
        self._medium_threshold = medium_threshold
        self._high_threshold = high_threshold
        self._anomaly_threshold = anomaly_threshold
        self._cooldown_frames = cooldown_frames

        # Frames restantes de silencio. 0 = listo para emitir.
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate(self, metrics: MotionMetrics) -> AlertEvent | None:
        """
        Evalúa las métricas del frame actual y devuelve un ``AlertEvent`` o ``None``.

        Devuelve ``None`` en tres casos:
            1. No hay movimiento (``metrics.motion_detected`` es False).
            2. El score no supera ``low_threshold``.
            3. El manager está en período de cooldown.

        Cuando emite una alerta, reinicia el cooldown a ``cooldown_frames``.

        Parámetros
        ----------
        metrics : MotionMetrics
            Métricas producidas por ``MotionAnalyzer.analyze()`` para el frame actual.

        Devuelve
        --------
        AlertEvent | None
        """
        # El cooldown se decrementa siempre, incluso sin movimiento,
        # para que el conteo de frames sea continuo y predecible.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if not metrics.motion_detected:
            return None

        severity = self._classify(metrics.anomaly_score)

        if severity is AlertSeverity.NONE:
            return None

        # Suprimir si aún estamos en cooldown
        if self._cooldown_remaining > 0:
            return None

        # Emitir alerta y reiniciar cooldown
        self._cooldown_remaining = self._cooldown_frames

        return AlertEvent(
            severity=severity,
            message=_message_for(severity),
            anomaly_score=metrics.anomaly_score,
            motion_ratio=metrics.motion_ratio,
            active_regions=metrics.active_regions,
            should_record=severity
            in {
                AlertSeverity.HIGH,
                AlertSeverity.ANOMALY,
            },
        )

    def reset(self) -> None:
        """
        Reinicia el cooldown a cero.

        Útil al cambiar de fuente de vídeo, reanudar una sesión pausada
        o en tests que necesiten que el manager esté listo para emitir
        desde la primera llamada a ``evaluate()``.
        """
        self._cooldown_remaining = 0

    # ------------------------------------------------------------------
    # Estado observable (read-only, útil para tests y HUD futuro)
    # ------------------------------------------------------------------

    @property
    def cooldown_remaining(self) -> int:
        """Frames de silencio que quedan antes de poder emitir otra alerta."""
        return self._cooldown_remaining

    @property
    def in_cooldown(self) -> bool:
        """``True`` si el manager está suprimiendo alertas por cooldown."""
        return self._cooldown_remaining > 0

    # ------------------------------------------------------------------
    # Clasificación interna
    # ------------------------------------------------------------------

    def _classify(self, score: float) -> AlertSeverity:
        """
        Mapea un ``anomaly_score`` al nivel de severidad correspondiente.

        Los umbrales se evalúan de mayor a menor para devolver siempre
        el nivel más alto que aplica.
        """
        if score >= self._anomaly_threshold:
            return AlertSeverity.ANOMALY
        if score >= self._high_threshold:
            return AlertSeverity.HIGH
        if score >= self._medium_threshold:
            return AlertSeverity.MEDIUM
        if score >= self._low_threshold:
            return AlertSeverity.LOW
        return AlertSeverity.NONE


# ---------------------------------------------------------------------------
# Funciones puras auxiliares
# ---------------------------------------------------------------------------


def _message_for(severity: AlertSeverity) -> str:
    """
    Devuelve el mensaje técnico canónico para cada nivel de severidad.

    Los mensajes son descriptivos del fenómeno observado, no interpretativos.
    Las capas superiores (notif.py, dashboard.py) pueden enriquecerlos con
    contexto de sesión o localización si es necesario.
    """
    return {
        AlertSeverity.LOW: "Poca actividad física",
        AlertSeverity.MEDIUM: "Actividad física de intensidad moderada",
        AlertSeverity.HIGH: "Alta actividad física",
        AlertSeverity.ANOMALY: "Posible anomalía visual",
    }[severity]


def _validate_thresholds(
    low: float, medium: float, high: float, anomaly: float
) -> None:
    """
    Verifica que los umbrales forman una secuencia estrictamente creciente
    en el intervalo (0, 1).

    Una mala configuración aquí produciría severidades inalcanzables o
    comportamiento ambiguo, por lo que fallamos rápido en el constructor.
    """
    values = [low, medium, high, anomaly]
    names = ["low_threshold", "medium_threshold", "high_threshold", "anomaly_threshold"]

    for name, value in zip(names, values, strict=True):
        if not 0.0 < value < 1.0:
            msg = f"{name}={value} debe estar en el intervalo abierto (0, 1)."
            raise ValueError(msg)

    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            msg = (
                f"{names[i]}={values[i]} debe ser estrictamente menor que "
                f"{names[i + 1]}={values[i + 1]}."
            )
            raise ValueError(msg)
