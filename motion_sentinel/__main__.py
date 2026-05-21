from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from motion_sentinel.capture.video_source import VideoSourceError
from motion_sentinel.common.app_config import ConfigOverrides, load_app_config
from motion_sentinel.common.logger import get_logger, setup_logging
from motion_sentinel.pipeline import SessionRunner

_CONFIG_PATH = Path("config/default.yaml")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the Motion Sentinel CLI parser."""
    parser = argparse.ArgumentParser(
        prog="motion-sentinel",
        description="Real-time motion detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to the base YAML config file.",
    )
    parser.add_argument(
        "--profile",
        help="Profile name from config/profiles or a direct YAML profile path.",
    )
    parser.add_argument(
        "--source",
        help="Override capture.source with a webcam index, file path, or RTSP/HTTP URL.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=None,
        help="Disable OpenCV windows and keyboard polling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override recorder.output_dir for snapshots.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. Kept separate for tests."""
    return build_arg_parser().parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> ConfigOverrides:
    return ConfigOverrides(
        source=args.source,
        headless=args.headless,
        output_dir=args.output_dir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        cfg = load_app_config(
            args.config,
            profile=args.profile,
            overrides=_overrides_from_args(args),
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Config error: {exc}\n")
        sys.exit(2)

    setup_logging(level=cfg.app.log_level)
    log = get_logger(__name__)

    log.info(
        "Inicializando Motion Sentinel",
        version=cfg.app.version,
        log_level=cfg.app.log_level,
    )
    log.info(
        "Config cargada",
        config=str(cfg.config_path) if cfg.config_path else None,
        profile=str(cfg.profile_path) if cfg.profile_path else None,
    )

    try:
        SessionRunner(cfg, log=log).run()
    except VideoSourceError as exc:
        log.error("No se ha podido abrir la fuente de vídeo", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
