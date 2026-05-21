import numpy as np

from motion_sentinel.common.app_config import AppConfig
from motion_sentinel.pipeline import SessionRunner


def test_headless_show_frame_does_not_call_opencv(monkeypatch) -> None:
    calls: list[str] = []

    def fake_imshow(*_args) -> None:
        calls.append("imshow")

    def fake_wait_key(*_args) -> int:
        calls.append("waitKey")
        return -1

    monkeypatch.setattr("motion_sentinel.pipeline.cv2.imshow", fake_imshow)
    monkeypatch.setattr("motion_sentinel.pipeline.cv2.waitKey", fake_wait_key)

    runner = SessionRunner(AppConfig(headless=True))
    should_exit = runner._show_frame("test", np.zeros((2, 2, 3), dtype=np.uint8))

    assert should_exit is False
    assert calls == []
