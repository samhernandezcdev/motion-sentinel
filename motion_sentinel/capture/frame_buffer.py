"""
Buffer thread-safe de frames con producer/consumer desacoplados.

Arquitectura:
    VideoSource (hilo productor)
        └─► FrameBuffer (queue thread-safe)
                └─► consumer loop en __main__ (hilo principal)

Solo usa stdlib: queue.Queue + threading.Thread + threading.Event.
Sin asyncio, sin multiprocessing, sin EventBus.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from dataclasses import dataclass, field

import numpy as np

from motion_sentinel.capture.video_source import VideoSource
from motion_sentinel.common.logger import get_logger

log = get_logger(__name__)


@dataclass
class FrameBufferConfig:
    """
    Parámetros del buffer de frames.

    Atributos
    ----------
    maxsize:
        Número máximo de frames almacenados simultáneamente.
        Un valor pequeño (4-16) es suficiente para desacoplar captura
        y procesamiento sin consumir mucha memoria.
    drop_oldest:
        Si ``True`` y el buffer está lleno, descarta el frame más antiguo
        antes de insertar el nuevo. Así el consumer siempre recibe frames
        frescos, a costa de perder algunos intermedios.
        Si ``False`` el productor espera hasta que haya hueco (modo lossless).
    """

    maxsize: int = 8
    drop_oldest: bool = True


class FrameBuffer:
    """
    Queue thread-safe de frames BGR (numpy.ndarray).

    Encapsula un ``queue.Queue`` y añade la política drop_oldest para
    evitar que el productor se bloquee cuando el consumer es más lento.

    Diseñado para un único productor y un único consumer (MVP).
    """

    def __init__(self, config: FrameBufferConfig | None = None) -> None:
        self._cfg = config or FrameBufferConfig()
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=self._cfg.maxsize)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def put(self, frame: np.ndarray) -> None:
        """
        Inserta un frame en el buffer.

        Si ``drop_oldest=True`` y el buffer está lleno, extrae y descarta
        el frame más antiguo antes de insertar, garantizando que el
        productor nunca se bloquee.

        Si ``drop_oldest=False`` bloquea al productor hasta que haya espacio
        (comportamiento estándar de ``queue.Queue``).
        """
        if self._cfg.drop_oldest and self._q.full():
            try:
                self._q.get_nowait()   # descartar el frame más viejo
            except queue.Empty:
                pass                   # race condition inocua; el consumer lo tomó

        try:
            self._q.put_nowait(frame)
        except queue.Full:
            # Solo alcanzable en modo lossless con alta contención;
            # se registra y se descarta silenciosamente.
            log.warning("FrameBuffer lleno, frame descartado")

    def get(self, timeout: float | None = None) -> np.ndarray | None:
        """
        Extrae el frame más antiguo del buffer.

        Parámetros
        ----------
        timeout:
            Segundos máximos de espera. ``None`` = bloquear indefinidamente.
            Se recomienda pasar un valor pequeño (p. ej. 0.1) en loops para
            poder comprobar señales de parada.

        Devuelve
        --------
        np.ndarray si hay un frame disponible, ``None`` si se agota el timeout.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Vacía el buffer descartando todos los frames pendientes."""
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def size(self) -> int:
        """Número de frames actualmente en el buffer."""
        return self._q.qsize()

    def empty(self) -> bool:
        """``True`` si el buffer está vacío."""
        return self._q.empty()


# ---------------------------------------------------------------------------
# BufferedVideoSource
# ---------------------------------------------------------------------------


@dataclass
class _ProducerState:
    """Estado interno mutable del hilo productor."""

    frames_captured: int = 0
    frames_dropped: int = 0
    running: bool = field(default=False, init=False)


class BufferedVideoSource:
    """
    Wrapper sobre :class:`VideoSource` que captura en un hilo separado.

    El hilo productor lee frames de la cámara continuamente y los empuja
    al :class:`FrameBuffer`. El consumer (loop principal) los extrae a su
    propio ritmo sin bloquear la captura.

    Parámetros
    ----------
    video_source:
        Instancia de :class:`VideoSource` ya construida (sin abrir todavía).
    buffer_config:
        Configuración del buffer. Si es ``None`` se usan los defaults.

    Uso típico::

        vs = VideoSource(source=0, width=1280, height=720)
        bvs = BufferedVideoSource(vs)
        bvs.start()

        for frame in bvs.frames():
            # procesar frame
            ...

        bvs.stop()
    """

    def __init__(
        self,
        video_source: VideoSource,
        buffer_config: FrameBufferConfig | None = None,
    ) -> None:
        self._source = video_source
        self._buffer = FrameBuffer(buffer_config)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = _ProducerState()

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Abre la fuente de vídeo y lanza el hilo productor.

        El hilo es ``daemon=True``: si el proceso principal termina, el
        hilo se cierra automáticamente sin necesidad de llamar a ``stop()``.
        """
        if self._thread is not None and self._thread.is_alive():
            log.warning("BufferedVideoSource ya está en ejecución")
            return

        self._stop_event.clear()
        self._buffer.clear()
        self._state = _ProducerState()

        self._source.open()

        self._thread = threading.Thread(
            target=self._producer_loop,
            name="frame-producer",
            daemon=True,
        )
        self._thread.start()
        log.info("Hilo productor iniciado", buffer_maxsize=self._buffer._cfg.maxsize)

    def stop(self) -> None:
        """
        Señala al hilo productor que se detenga y espera su finalización.

        Seguro de llamar múltiples veces o aunque ``start()`` no se haya
        invocado todavía.
        """
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("El hilo productor no terminó en el tiempo esperado")
            self._thread = None

        self._source.close()
        log.info(
            "BufferedVideoSource detenido",
            frames_capturados=self._state.frames_captured,
            frames_descartados=self._state.frames_dropped,
        )

    # ------------------------------------------------------------------
    # Iteración para el consumer
    # ------------------------------------------------------------------

    def frames(self) -> Generator[np.ndarray, None, None]:
        """
        Generador que produce frames desde el buffer.

        Bloquea hasta 100 ms en cada intento de ``get``. Si el buffer está
        vacío y el hilo productor ya terminó (fuente agotada o ``stop()``
        llamado), el generador finaliza limpiamente.
        """
        while True:
            # Comprobar señal de parada solo si el buffer ya está vacío
            frame = self._buffer.get(timeout=0.1)

            if frame is None:
                # Buffer vacío: comprobar si el productor ya terminó
                if self._stop_event.is_set() and self._buffer.empty():
                    break
                continue  # buffer momentáneamente vacío; reintentar

            yield frame

    # ------------------------------------------------------------------
    # Hilo productor (privado)
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """
        Loop interno del hilo de captura.

        Lee frames de :class:`VideoSource` y los empuja al buffer hasta
        que ``_stop_event`` se activa o la fuente se agota.
        """
        log.debug("Hilo productor arrancado")

        try:
            for frame in self._source.frames():
                if self._stop_event.is_set():
                    break

                before = self._buffer.size()
                self._buffer.put(frame)
                after = self._buffer.size()

                self._state.frames_captured += 1

                # Si el tamaño no creció, se descartó el frame más viejo
                if after <= before and before == self._buffer._cfg.maxsize:
                    self._state.frames_dropped += 1

        except Exception:
            log.exception("Error inesperado en el hilo productor")
        finally:
            self._stop_event.set()   # notificar al consumer que terminamos
            log.debug(
                "Hilo productor finalizado",
                capturados=self._state.frames_captured,
                descartados=self._state.frames_dropped,
            )
