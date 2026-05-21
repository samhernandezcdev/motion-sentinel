"""Helpers for working with video source URLs."""

from __future__ import annotations

from urllib.parse import SplitResult, urlsplit, urlunsplit

_NETWORK_VIDEO_SCHEMES = {"rtsp", "rtsps", "http", "https"}


def is_network_video_source(source: int | str) -> bool:
    """Return True for RTSP/HTTP-style video sources."""
    if not isinstance(source, str):
        return False
    return urlsplit(source).scheme.lower() in _NETWORK_VIDEO_SCHEMES


def is_rewindable_local_file(source: int | str) -> bool:
    """Return True when a string source should be treated as a local video file."""
    if not isinstance(source, str):
        return False

    parsed = urlsplit(source)
    scheme = parsed.scheme.lower()

    if scheme in _NETWORK_VIDEO_SCHEMES:
        return False

    return scheme in {"", "file"}


def redact_url_credentials(source: int | str) -> int | str:
    """Redact username/password credentials from loggable video source URLs."""
    if not isinstance(source, str):
        return source

    parsed = urlsplit(source)
    if parsed.scheme.lower() not in _NETWORK_VIDEO_SCHEMES:
        return source

    if "@" not in parsed.netloc:
        return source

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"

    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        port = ""

    redacted = SplitResult(
        scheme=parsed.scheme,
        netloc=f"***:***@{host}{port}",
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
    )
    return urlunsplit(redacted)
