"""
Sistema de zonas de observación rectangular (ROI = Region of Interest).

No filtra ni descarta movimiento. Responde a la pregunta:
"¿En qué zonas configuradas está ocurriendo movimiento, y cuánto?"

Flujo de uso::

    manager = ROIManager()
    manager.add_roi(ROI("puerta",   x=0,   y=100, width=200, height=300))
    manager.add_roi(ROI("ventana",  x=400, y=50,  width=150, height=200, weight=1.5))

    hits = manager.analyze_regions(regions, frame_shape=(720, 1280))
    for hit in hits:
        # hit.roi_name, hit.region_count, hit.total_area, hit.weighted_score
        ...
"""

from __future__ import annotations

from dataclasses import dataclass

from motion_sentinel.core.motion_detector import MotionRegion


@dataclass(frozen=True)
class ROI:
    """
    Zona de observación rectangular sobre el frame.

    Atributos
    ----------
    name:
        Identificador legible para logs y configuración.
    x, y:
        Coordenadas de la esquina superior-izquierda (píxeles).
    width, height:
        Dimensiones del rectángulo (píxeles).
    weight:
        Factor de importancia. Un ROI con ``weight=2.0`` duplica su
        ``weighted_score`` respecto a uno con ``weight=1.0`` de igual área.
        Útil para priorizar zonas sensibles (caja fuerte, puerta principal).
    enabled:
        Si es ``False``, el ROI se ignora en el análisis.
    """

    name: str
    x: int
    y: int
    width: int
    height: int
    weight: float = 1.0
    enabled: bool = True

    # ------------------------------------------------------------------
    # Propiedades derivadas
    # ------------------------------------------------------------------

    @property
    def x2(self) -> int:
        """Coordenada X del borde derecho (exclusiva)."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Coordenada Y del borde inferior (exclusiva)."""
        return self.y + self.height

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def contains_point(self, px: int, py: int) -> bool:
        """
        Devuelve ``True`` si ``(px, py)`` está dentro del rectángulo.

        El borde superior-izquierdo es inclusivo; el inferior-derecho,
        exclusivo — igual que los slices de numpy.
        """
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def clamp(self, frame_h: int, frame_w: int) -> ROI:
        """
        Devuelve una copia del ROI recortada a los límites del frame.

        Garantiza que ninguna coordenada supere la resolución real de la
        cámara, evitando errores de indexación cuando la config tiene
        valores mayores que la resolución actual.
        """
        cx = max(0, min(self.x, frame_w))
        cy = max(0, min(self.y, frame_h))
        cx2 = max(0, min(self.x2, frame_w))
        cy2 = max(0, min(self.y2, frame_h))
        return ROI(
            name=self.name,
            x=cx,
            y=cy,
            width=cx2 - cx,
            height=cy2 - cy,
            weight=self.weight,
            enabled=self.enabled,
        )

    def is_valid(self) -> bool:
        """``True`` si el ROI tiene área positiva (width > 0 y height > 0)."""
        return self.width > 0 and self.height > 0


@dataclass(frozen=True)
class ROIHit:
    """
    Resultado del análisis de una zona de observación para un frame.

    Atributos
    ----------
    roi_name:
        Nombre del ROI que generó este resultado.
    region_count:
        Número de MotionRegions cuyo centroide cae dentro del ROI.
    total_area:
        Suma de áreas (px²) de todas las regiones observadas en el ROI.
    weighted_score:
        ``total_area * roi.weight``. Métrica comparativa entre ROIs que
        tiene en cuenta la importancia relativa de cada zona.
    """

    roi_name: str
    region_count: int
    total_area: float
    weighted_score: float


class ROIManager:
    """
    Gestiona un conjunto de ROIs y analiza en cuáles hay actividad.

    No modifica regiones ni máscaras. Solo observa y acumula métricas
    por zona para que el resto del pipeline pueda actuar sobre ellas.
    """

    def __init__(self) -> None:
        self._rois: list[ROI] = []

    # ------------------------------------------------------------------
    # Gestión de ROIs
    # ------------------------------------------------------------------

    def add_roi(self, roi: ROI) -> None:
        """Registra un nuevo ROI. Se permiten nombres duplicados."""
        self._rois.append(roi)

    def remove_roi(self, name: str) -> None:
        """Elimina todos los ROIs con el nombre indicado."""
        self._rois = [r for r in self._rois if r.name != name]

    def clear(self) -> None:
        """Elimina todos los ROIs registrados."""
        self._rois.clear()

    def enabled_rois(self) -> list[ROI]:
        """Devuelve los ROIs con ``enabled=True``."""
        return [r for r in self._rois if r.enabled]

    # ------------------------------------------------------------------
    # Análisis
    # ------------------------------------------------------------------

    def analyze_regions(
        self,
        regions: list[MotionRegion],
        frame_shape: tuple[int, int],
    ) -> list[ROIHit]:
        """
        Determina en qué ROIs hay actividad y acumula métricas por zona.

        La pertenencia de una :class:`MotionRegion` a un ROI se evalúa
        por el centroide de su bounding box: si el centroide cae dentro
        del rectángulo del ROI, la región se contabiliza en él.

        Una región puede pertenecer a más de un ROI si se solapan.
        No se modifica ninguna región ni máscara.

        Parámetros
        ----------
        regions:
            Lista de :class:`MotionRegion` devuelta por el detector.
            Puede estar vacía; en ese caso se devuelve ``[]``.
        frame_shape:
            ``(height, width)`` del frame procesado. Se usa para recortar
            los ROIs al tamaño real antes de evaluar pertenencia.

        Devuelve
        --------
        list[ROIHit]
            Un :class:`ROIHit` por cada ROI activo que tenga al menos una
            región dentro. ROIs sin actividad no aparecen en el resultado.
            La lista está ordenada de mayor a menor ``weighted_score``.
        """
        active = self.enabled_rois()
        if not active or not regions:
            return []

        frame_h, frame_w = frame_shape

        # Acumuladores indexados por roi.name
        counts: dict[str, int] = {}
        areas: dict[str, float] = {}
        weights: dict[str, float] = {}

        for roi in active:
            safe = roi.clamp(frame_h, frame_w)
            if not safe.is_valid():
                continue

            for region in regions:
                cx, cy = _centroid(region)
                if safe.contains_point(cx, cy):
                    counts[roi.name] = counts.get(roi.name, 0) + 1
                    areas[roi.name] = areas.get(roi.name, 0.0) + region.area
                    weights[roi.name] = roi.weight  # constante por ROI

        hits = [
            ROIHit(
                roi_name=name,
                region_count=counts[name],
                total_area=areas[name],
                weighted_score=areas[name] * weights[name],
            )
            for name in counts
        ]

        hits.sort(key=lambda h: h.weighted_score, reverse=True)
        return hits


# ---------------------------------------------------------------------------
# Helper interno
# ---------------------------------------------------------------------------


def _centroid(region: MotionRegion) -> tuple[int, int]:
    """Centroide entero del bounding box de una MotionRegion."""
    x, y, w, h = region.bbox
    return x + w // 2, y + h // 2
