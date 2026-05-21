import numpy as np

from motion_sentinel.core.motion_detector import FrameDifferenceMotionDetector
from motion_sentinel.core.preprocessor import preprocess_for_motion


def test_detector_first_frame_empty_then_detects_change() -> None:
    detector = FrameDifferenceMotionDetector(
        threshold=10,
        min_area=20,
        dilation_iterations=1,
    )

    first = np.zeros((100, 100, 3), dtype=np.uint8)
    second = first.copy()
    second[30:70, 30:70] = 255

    processed_first = preprocess_for_motion(first, resize_width=100, blur_kernel=3)
    mask1, regions1 = detector.detect(processed_first)

    assert mask1.sum() == 0
    assert regions1 == []

    processed_second = preprocess_for_motion(second, resize_width=100, blur_kernel=3)
    mask2, regions2 = detector.detect(processed_second)

    assert mask2.sum() > 0
    assert regions2
    assert regions2[0].area > 0
