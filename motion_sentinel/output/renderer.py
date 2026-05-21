"""
Renderiza overlay visual sobre el frame BGR original.

Dibuja bounding boxes de cada MotionRegion y un HUD con métricas de sesión.
No modifica el frame original: opera sobre una copia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from motion_sentinel.core.motion_detector import MotionRegion
from motion_sentinel.core.roi_manager import ROI, ROIHit

if TYPE_CHECKING:
    from motion_sentinel.analysis.motion_analyzer import MotionMetrics

# ---------------------------------------------------------------------------
# Paleta de colores (BGR)
# ---------------------------------------------------------------------------

_GREEN = (0, 220, 0)
_YELLOW = (0, 200, 220)
_RED = (0, 0, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GRAY = (160, 160, 160)
_OVERLAY_BG = (20, 20, 20)
_CYAN = (200, 200, 0)  # ROI activo
_DIM_CYAN = (100, 100, 0)  # ROI sin actividad


# ---------------------------------------------------------------------------
# Configuración del renderer
# ---------------------------------------------------------------------------


@dataclass
class RendererConfig:
    """
    Parámetros visuales del renderer.

    Todos los valores tienen defaults razonables para no requerir configuración
    explícita en el MVP.
    """

    bbox_color_motion: tuple[int, int, int] = _GREEN
    bbox_color_idle: tuple[int, int, int] = _GRAY
    bbox_thickness: int = 2
    hud_font: int = cv2.FONT_HERSHEY_SIMPLEX
    hud_font_scale: float = 0.55
    hud_font_thickness: int = 1
    hud_margin: int = 10  # px desde el borde superior-izquierdo
    hud_line_height: int = 22  # px entre líneas del HUD
    hud_bg_alpha: float = 0.45  # opacidad del fondo semitransparente del HUD
    show_mask_inset: bool = False  # miniatura de la máscara en esquina inferior
    show_rois: bool = True  # dibujar zonas ROI sobre el frame
    roi_thickness: int = 1  # grosor de línea de los rectángulos ROI
    anomaly_alert_threshold: float = (
        0.35  # umbral de anomaly_score a partir del cual el score se muestra en rojo
    )


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------


def render_motion_overlay(
    frame: np.ndarray,
    regions: list[MotionRegion],
    mask: np.ndarray | None = None,
    metrics: MotionMetrics | None = None,
    config: RendererConfig | None = None,
    rois: list[ROI] | None = None,
    roi_hits: list[ROIHit] | None = None,
) -> np.ndarray:
    """
    Dibuja bounding boxes y HUD sobre una copia del frame.

    Parámetros
    ----------
    frame:
        Frame BGR original (H, W, 3) uint8. No se modifica.
    regions:
        Lista de :class:`MotionRegion` devuelta por el detector.
    mask:
        Máscara binaria opcional (H, W) uint8. Si se pasa y
        ``config.show_mask_inset`` es True, se muestra como miniatura.
    metrics:
        Instancia de :class:`~motion_sentinel.analysis.motion_analyzer.MotionMetrics`
        producida por ```MotionAnalyzer.analyze()``. si se pasa, el HUD muestra
        las cinco métricas analíticas; si es None, el HUD cae al modo legado
        (solo estado, regiones, y área total).
    config:
        Opciones visuales. Si es None se usan los defaults de RendererConfig.
    rois:
        Lista de :class:`ROI` activos a dibujar. Si es None o lista vacía,
        no se dibuja ninguna zona.
    roi_hits:
        Hits del frame actual devueltos por ``ROIManager.analyze_regions``.
        Si se pasan, los ROIs con actividad se resaltan con métricas inline.

    Devuelve
    --------
    np.ndarray
        Frame BGR con el overlay aplicado.
    """
    if config is None:
        config = RendererConfig()

    output = frame.copy()
    motion_detected = len(regions) > 0

    # Los ROIs se dibujan primero para que los bboxes queden encima
    if config.show_rois and rois:
        _draw_rois(output, rois, roi_hits or [], config)

    _draw_bboxes(output, regions, config, motion_detected)
    _draw_hud(output, regions, config, motion_detected, metrics)

    if mask is not None and config.show_mask_inset:
        _draw_mask_inset(output, mask)

    return output


# ---------------------------------------------------------------------------
# Helpers de dibujo
# ---------------------------------------------------------------------------


def _draw_rois(
    frame: np.ndarray,
    rois: list[ROI],
    roi_hits: list[ROIHit],
    config: RendererConfig,
) -> None:
    """
    Dibuja cada ROI activo sobre el frame.

    ROIs con actividad (presentes en ``roi_hits``) se dibujan en cyan
    brillante con métricas inline (área y score). ROIs sin actividad se
    dibujan en cyan tenue con solo nombre y weight.
    """
    # Índice rápido para saber qué ROIs tienen hits este frame
    hits_by_name: dict[str, ROIHit] = {h.roi_name: h for h in roi_hits}

    font = config.hud_font
    scale = config.hud_font_scale * 0.80
    thick = config.hud_font_thickness

    for roi in rois:
        hit = hits_by_name.get(roi.name)
        active = hit is not None
        color = _CYAN if active else _DIM_CYAN

        # Rectángulo del ROI
        cv2.rectangle(
            frame,
            (roi.x, roi.y),
            (roi.x2, roi.y2),
            color,
            config.roi_thickness,
        )

        # Etiqueta: "nombre  x<weight>" en la esquina superior del ROI
        weight_str = f" x{roi.weight:.1g}" if roi.weight != 1.0 else ""
        label = f"{roi.name}{weight_str}"

        label_y = max(roi.y - 4, 10)
        cv2.putText(
            frame,
            label,
            (roi.x + 2, label_y),
            font,
            scale,
            _BLACK,
            thick + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, label, (roi.x + 2, label_y), font, scale, color, thick, cv2.LINE_AA
        )

        # Métricas inline solo si hay actividad
        if active and hit is not None:
            metrics_str = f"area={int(hit.total_area)} sc={hit.weighted_score:.0f}"
            metrics_y = min(roi.y2 + 12, frame.shape[0] - 4)
            cv2.putText(
                frame,
                metrics_str,
                (roi.x + 2, metrics_y),
                font,
                scale,
                _BLACK,
                thick + 1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                metrics_str,
                (roi.x + 2, metrics_y),
                font,
                scale,
                color,
                thick,
                cv2.LINE_AA,
            )


def _draw_bboxes(
    frame: np.ndarray,
    regions: list[MotionRegion],
    config: RendererConfig,
    motion_detected: bool,
) -> None:
    """Dibuja un rectángulo por cada MotionRegion."""
    color = config.bbox_color_motion if motion_detected else config.bbox_color_idle

    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, config.bbox_thickness)

        # Etiqueta pequeña con el área de la región
        label = f"{int(region.area)} px^2"
        cv2.putText(
            frame,
            label,
            (x, max(y - 6, 10)),
            config.hud_font,
            config.hud_font_scale * 0.85,
            color,
            config.hud_font_thickness,
            cv2.LINE_AA,
        )


def _draw_hud(
    frame: np.ndarray,
    regions: list[MotionRegion],
    config: RendererConfig,
    motion_detected: bool,
    metrics: MotionMetrics | None,
) -> None:
    """
    Dibuja el HUD en la esquina superior izquierda.

    Con ``metrics``:
        • Estado         : Movimiento detectado / Sin movimiento
        • Regiones       : active_regions
        • Área total     : total_motion_area  (px²)
        • Ratio          : motion_ratio       (porcentaje del frame)
        • Anomaly score  : anomaly_score      (0.00 - 1.00, coloreado)

    Sin ``metrics`` (modo legado):
        • Estado / Regiones / Área total  (comportamiento original)
    """
    status_text = "Movimiento detectado" if motion_detected else "Sin movimiento"
    status_color = _GREEN if motion_detected else _GRAY

    if metrics is not None:
        lines = _build_metrics_lines(metrics, status_text, status_color, config)
    else:
        total_area = sum(r.area for r in regions)

        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"Estado   : {status_text}", status_color),
            (f"Regiones : {len(regions)}", _WHITE),
            (f"Area     : {int(total_area)} px^2", _WHITE),
        ]

    _render_hud_lines(frame, lines, config)


def _build_metrics_lines(
    metrics: MotionMetrics,
    status_text: str,
    status_color: tuple[int, int, int],
    config: RendererConfig,
) -> list[tuple[str, tuple[int, int, int]]]:
    """
    Construye la lista de (texto, color) para el HUD analítico.

    El anomaly_score se colorea según ``config.anomaly_alert_threshold``:
        • score < threshold   → blanco   (actividad normal)
        • threshold ≤ score < 0.65 → amarillo (actividad elevada)
        • score ≥ 0.65        → rojo     (anomalía marcada)
    """
    score = metrics.anomaly_score
    threshold = config.anomaly_alert_threshold

    if score >= 0.65:  # noqa: PLR2004
        score_color = _RED
    elif score >= threshold:
        score_color = _YELLOW
    else:
        score_color = _WHITE

    return [
        (f"Estado   : {status_text}", status_color),
        (f"Regiones : {metrics.active_regions}", _WHITE),
        (f"Area     : {int(metrics.total_motion_area)} px^2", _WHITE),
        (f"Tasa     : {metrics.motion_ratio:.4f}", _WHITE),
        (f"Anomalia : {score:.4f}", score_color),
    ]


def _render_hud_lines(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    config: RendererConfig,
) -> None:
    """Renderiza las líneas del HUD con fondo semitransparente."""
    m = config.hud_margin
    lh = config.hud_line_height
    font = config.hud_font
    scale = config.hud_font_scale
    thick = config.hud_font_thickness

    # Calcular el ancho máximo de texto para el fondo semitransparente
    text_widths = [cv2.getTextSize(text, font, scale, thick)[0][0] for text, _ in lines]
    bg_w = max(text_widths) + m * 2
    bg_h = lh * len(lines) + m

    # Fondo semitransparente (blend sobre ROI)
    roi = frame[m : m + bg_h, m : m + bg_w]
    if roi.size > 0:
        bg = np.full_like(roi, _OVERLAY_BG)
        cv2.addWeighted(bg, config.hud_bg_alpha, roi, 1 - config.hud_bg_alpha, 0, roi)
        frame[m : m + bg_h, m : m + bg_w] = roi

    # Texto línea a línea
    for i, (text, color) in enumerate(lines):
        y_pos = m + lh * (i + 1) - 4
        cv2.putText(
            frame, text, (m + 4, y_pos), font, scale, _BLACK, thick + 1, cv2.LINE_AA
        )
        cv2.putText(frame, text, (m + 4, y_pos), font, scale, color, thick, cv2.LINE_AA)


def _draw_mask_inset(frame: np.ndarray, mask: np.ndarray) -> None:
    """
    Dibuja una miniatura de la máscara binaria en la esquina inferior derecha.

    La miniatura ocupa 1/5 del ancho del frame.
    """
    h, w = frame.shape[:2]
    inset_w = w // 5
    inset_h = h // 5

    # Convertir máscara a BGR para poder pegarla sobre el frame
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    inset = cv2.resize(mask_bgr, (inset_w, inset_h), interpolation=cv2.INTER_AREA)

    y1, y2 = h - inset_h - 10, h - 10
    x1, x2 = w - inset_w - 10, w - 10
    frame[y1:y2, x1:x2] = inset

    # Borde alrededor de la miniatura
    cv2.rectangle(frame, (x1, y1), (x2, y2), _WHITE, 1)
