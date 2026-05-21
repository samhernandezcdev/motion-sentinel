from motion_sentinel.common.url_utils import (
    is_network_video_source,
    is_rewindable_local_file,
    redact_url_credentials,
)


def test_rtsp_credentials_are_redacted_and_not_rewindable() -> None:
    source = "rtsp://user:pass@example.test:554/live?channel=1"

    assert redact_url_credentials(source) == (
        "rtsp://***:***@example.test:554/live?channel=1"
    )
    assert is_network_video_source(source) is True
    assert is_rewindable_local_file(source) is False


def test_local_file_sources_are_rewindable() -> None:
    assert is_network_video_source("data/test_video.mp4") is False
    assert is_rewindable_local_file("data/test_video.mp4") is True
