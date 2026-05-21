import threading

import numpy as np

from motion_sentinel.capture.frame_buffer import FrameBuffer, FrameBufferConfig


def _frame(value: int) -> np.ndarray:
    return np.full((1, 1, 3), value, dtype=np.uint8)


def test_drop_oldest_keeps_freshest_frame() -> None:
    buffer = FrameBuffer(FrameBufferConfig(maxsize=1, drop_oldest=True))

    assert buffer.put(_frame(1)) is True
    assert buffer.put(_frame(2)) is True

    frame = buffer.get(timeout=0.1)
    assert frame is not None
    assert int(frame[0, 0, 0]) == 2


def test_lossless_mode_blocks_until_space_is_available() -> None:
    buffer = FrameBuffer(FrameBufferConfig(maxsize=1, drop_oldest=False))
    inserted = threading.Event()

    assert buffer.put(_frame(1)) is True

    def put_second_frame() -> None:
        buffer.put(_frame(2))
        inserted.set()

    thread = threading.Thread(target=put_second_frame)
    thread.start()

    assert inserted.wait(timeout=0.05) is False

    first = buffer.get(timeout=0.1)
    assert first is not None
    assert int(first[0, 0, 0]) == 1

    thread.join(timeout=1.0)
    assert inserted.is_set()

    second = buffer.get(timeout=0.1)
    assert second is not None
    assert int(second[0, 0, 0]) == 2
