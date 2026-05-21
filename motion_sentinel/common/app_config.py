"""Typed application configuration for Motion Sentinel."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

_INT_SOURCE_RE = re.compile(r"^[+-]?\d+$")


@dataclass(frozen=True)
class AppSettings:
    name: str = "motion-sentinel"
    version: str = "0.1.0"
    log_level: str = "INFO"


@dataclass(frozen=True)
class CaptureSettings:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    window_title: str = "Motion Sentinel"


@dataclass(frozen=True)
class DetectionSettings:
    resize_width: int | None = 640
    blur_kernel: int = 21
    threshold: int = 25
    min_area: float = 500.0
    dilation_iterations: int = 2


@dataclass(frozen=True)
class AnalyzerSettings:
    history_size: int = 30
    anomaly_weight_ratio: float = 0.6


@dataclass(frozen=True)
class RendererSettings:
    show_mask_inset: bool = False
    show_rois: bool = True


@dataclass(frozen=True)
class AlertSettings:
    low_threshold: float = 0.05
    medium_threshold: float = 0.15
    high_threshold: float = 0.30
    anomaly_threshold: float = 0.50
    cooldown_frames: int = 15


@dataclass(frozen=True)
class RecorderSettings:
    output_dir: Path = Path("data/snapshots")
    image_format: str = "jpg"
    jpeg_quality: int = 90
    filename_prefix: str = "event"


@dataclass(frozen=True)
class FrameBufferSettings:
    enabled: bool = False
    maxsize: int = 8
    drop_oldest: bool = True


@dataclass(frozen=True)
class ROISettings:
    name: str
    x: int
    y: int
    width: int
    height: int
    weight: float = 1.0
    enabled: bool = True


@dataclass(frozen=True)
class ConfigOverrides:
    source: int | str | None = None
    headless: bool | None = None
    output_dir: Path | str | None = None


@dataclass(frozen=True)
class AppConfig:
    app: AppSettings = AppSettings()
    capture: CaptureSettings = CaptureSettings()
    detection: DetectionSettings = DetectionSettings()
    analyzer: AnalyzerSettings = AnalyzerSettings()
    renderer: RendererSettings = RendererSettings()
    alerts: AlertSettings = AlertSettings()
    recorder: RecorderSettings = RecorderSettings()
    frame_buffer: FrameBufferSettings = FrameBufferSettings()
    rois: tuple[ROISettings, ...] = ()
    headless: bool = False
    config_path: Path | None = None
    profile_path: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        data: dict[str, Any],
        *,
        config_path: Path | None = None,
        profile_path: Path | None = None,
    ) -> AppConfig:
        app = _dataclass_from_section(AppSettings, data.get("app", {}))
        capture = _build_capture(data.get("capture", {}))
        detection = _build_detection(data.get("detection", {}))
        analyzer = _build_analyzer(data.get("analyzer", {}))
        renderer = _build_renderer(data.get("renderer", {}))
        alerts = _build_alerts(data.get("alerts", {}))
        recorder = _build_recorder(data.get("recorder", {}))
        frame_buffer = _build_frame_buffer(data.get("frame_buffer", {}))
        rois = tuple(_build_roi(entry) for entry in data.get("rois", []) or [])
        headless = _as_bool(data.get("headless", False), "headless")

        return cls(
            app=app,
            capture=capture,
            detection=detection,
            analyzer=analyzer,
            renderer=renderer,
            alerts=alerts,
            recorder=recorder,
            frame_buffer=frame_buffer,
            rois=rois,
            headless=headless,
            config_path=config_path,
            profile_path=profile_path,
        )

    def with_overrides(self, overrides: ConfigOverrides | None) -> AppConfig:
        if overrides is None:
            return self

        cfg = self

        if overrides.source is not None:
            cfg = replace(
                cfg,
                capture=replace(
                    cfg.capture, source=_normalize_source(overrides.source)
                ),
            )

        if overrides.headless is not None:
            cfg = replace(cfg, headless=overrides.headless)

        if overrides.output_dir is not None:
            cfg = replace(
                cfg,
                recorder=replace(cfg.recorder, output_dir=Path(overrides.output_dir)),
            )

        return cfg


def load_app_config(
    config_path: str | Path = Path("config/default.yaml"),
    *,
    profile: str | Path | None = None,
    overrides: ConfigOverrides | None = None,
) -> AppConfig:
    """Load base YAML, optionally merge a profile, and return typed config."""
    base_path = Path(config_path)
    data = _read_yaml_mapping(base_path)

    profile_path = None
    if profile:
        profile_path = _resolve_profile_path(base_path, profile)
        data = _deep_merge(data, _read_yaml_mapping(profile_path))

    return AppConfig.from_mapping(
        data,
        config_path=base_path,
        profile_path=profile_path,
    ).with_overrides(overrides)


def _build_capture(section: dict[str, Any]) -> CaptureSettings:
    cfg = _dataclass_from_section(CaptureSettings, section)
    cfg = replace(cfg, source=_normalize_source(cfg.source))
    _validate_positive_int(cfg.width, "capture.width")
    _validate_positive_int(cfg.height, "capture.height")
    _validate_positive_int(cfg.fps, "capture.fps")
    return cfg


def _build_detection(section: dict[str, Any]) -> DetectionSettings:
    cfg = _dataclass_from_section(DetectionSettings, section)
    if cfg.resize_width is not None:
        _validate_positive_int(cfg.resize_width, "detection.resize_width")
    _validate_positive_int(cfg.blur_kernel, "detection.blur_kernel")
    _validate_range(cfg.threshold, "detection.threshold", 0, 255)
    if cfg.min_area < 0:
        raise ValueError("detection.min_area must be >= 0.")
    if cfg.dilation_iterations < 0:
        raise ValueError("detection.dilation_iterations must be >= 0.")
    return cfg


def _build_analyzer(section: dict[str, Any]) -> AnalyzerSettings:
    cfg = _dataclass_from_section(AnalyzerSettings, section)
    _validate_positive_int(cfg.history_size, "analyzer.history_size")
    _validate_open_unit(cfg.anomaly_weight_ratio, "analyzer.anomaly_weight_ratio")
    return cfg


def _build_renderer(section: dict[str, Any]) -> RendererSettings:
    return _dataclass_from_section(RendererSettings, section)


def _build_alerts(section: dict[str, Any]) -> AlertSettings:
    cfg = _dataclass_from_section(AlertSettings, section)
    values = [
        cfg.low_threshold,
        cfg.medium_threshold,
        cfg.high_threshold,
        cfg.anomaly_threshold,
    ]
    names = [
        "alerts.low_threshold",
        "alerts.medium_threshold",
        "alerts.high_threshold",
        "alerts.anomaly_threshold",
    ]

    for name, value in zip(names, values, strict=True):
        _validate_open_unit(value, name)

    for i in range(len(values) - 1):
        if values[i] >= values[i + 1]:
            raise ValueError(f"{names[i]} must be lower than {names[i + 1]}.")

    if cfg.cooldown_frames < 0:
        raise ValueError("alerts.cooldown_frames must be >= 0.")
    return cfg


def _build_recorder(section: dict[str, Any]) -> RecorderSettings:
    cfg = _dataclass_from_section(RecorderSettings, section)
    cfg = replace(cfg, output_dir=Path(cfg.output_dir))
    if not cfg.image_format:
        raise ValueError("recorder.image_format cannot be empty.")
    _validate_range(cfg.jpeg_quality, "recorder.jpeg_quality", 0, 100)
    if not cfg.filename_prefix:
        raise ValueError("recorder.filename_prefix cannot be empty.")
    return cfg


def _build_frame_buffer(section: dict[str, Any]) -> FrameBufferSettings:
    cfg = _dataclass_from_section(FrameBufferSettings, section)
    _validate_positive_int(cfg.maxsize, "frame_buffer.maxsize")
    return cfg


def _build_roi(entry: dict[str, Any]) -> ROISettings:
    if not isinstance(entry, dict):
        raise ValueError(f"ROI entry must be a mapping, got {type(entry).__name__}.")
    roi = _dataclass_from_section(ROISettings, entry)
    if not roi.name:
        raise ValueError("ROI name cannot be empty.")
    if roi.width <= 0 or roi.height <= 0:
        raise ValueError(f"ROI {roi.name!r} must have positive width and height.")
    if roi.weight <= 0:
        raise ValueError(f"ROI {roi.name!r} weight must be > 0.")
    return roi


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _resolve_profile_path(config_path: Path, profile: str | Path) -> Path:
    profile_path = Path(profile)
    profile_text = str(profile)

    is_direct_path = (
        profile_path.is_absolute()
        or profile_path.suffix
        or "/" in profile_text
        or "\\" in profile_text
    )

    if is_direct_path:
        return profile_path

    return config_path.parent / "profiles" / f"{profile_text}.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _dataclass_from_section(cls: type[Any], section: dict[str, Any]) -> Any:
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError(f"{cls.__name__} config section must be a mapping.")
    allowed = set(cls.__dataclass_fields__)
    filtered = {key: value for key, value in section.items() if key in allowed}
    return cls(**filtered)


def _normalize_source(source: int | str) -> int | str:
    if isinstance(source, int):
        return source
    if isinstance(source, str):
        stripped = source.strip()
        if _INT_SOURCE_RE.match(stripped):
            return int(stripped)
        return source
    raise ValueError("capture.source must be an int or string.")


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean.")


def _validate_positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_range(value: int, name: str, low: int, high: int) -> None:
    if not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}].")


def _validate_open_unit(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in the open interval (0, 1).")
