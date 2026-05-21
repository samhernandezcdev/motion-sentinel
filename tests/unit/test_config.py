from pathlib import Path

from motion_sentinel.__main__ import _overrides_from_args, parse_args
from motion_sentinel.common.app_config import load_app_config


def test_current_default_yaml_loads() -> None:
    cfg = load_app_config(Path("config/default.yaml"))

    assert cfg.capture.source == 0
    assert cfg.capture.width == 1280
    assert cfg.detection.resize_width == 640
    assert cfg.recorder.image_format == "jpg"
    assert cfg.frame_buffer.enabled is True
    assert len(cfg.rois) == 3


def test_profile_merge_and_cli_overrides(tmp_path: Path) -> None:
    base = tmp_path / "default.yaml"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    profile = profile_dir / "dev.yaml"

    base.write_text(
        """
capture:
  source: 0
  width: 1280
  height: 720
  fps: 30
recorder:
  output_dir: data/snapshots
frame_buffer:
  enabled: false
""",
        encoding="utf-8",
    )
    profile.write_text(
        """
capture:
  fps: 12
frame_buffer:
  enabled: true
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "snapshots"
    args = parse_args(
        [
            "--config",
            str(base),
            "--profile",
            "dev",
            "--source",
            "rtsp://user:pass@example.test/live",
            "--headless",
            "--output-dir",
            str(output_dir),
        ]
    )

    cfg = load_app_config(
        args.config,
        profile=args.profile,
        overrides=_overrides_from_args(args),
    )

    assert cfg.capture.source == "rtsp://user:pass@example.test/live"
    assert cfg.capture.fps == 12
    assert cfg.frame_buffer.enabled is True
    assert cfg.headless is True
    assert cfg.recorder.output_dir == output_dir


def test_unknown_future_keys_are_ignored_for_compatibility(tmp_path: Path) -> None:
    config = tmp_path / "default.yaml"
    config.write_text(
        """
capture:
  source: 0
  width: 640
  height: 480
  fps: 30
  reconnect_delay_s: 1.5
future_section:
  enabled: true
""",
        encoding="utf-8",
    )

    cfg = load_app_config(config)

    assert cfg.capture.source == 0
    assert cfg.capture.width == 640
