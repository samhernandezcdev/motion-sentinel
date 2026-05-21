from motion_sentinel.core.motion_detector import MotionRegion
from motion_sentinel.core.roi_manager import ROI, ROIManager


def test_roi_scales_from_source_to_processed_shape() -> None:
    roi = ROI(name="ventana", x=850, y=80, width=200, height=180, weight=1.5)

    scaled = roi.scaled(from_shape=(720, 1280), to_shape=(360, 640))

    assert scaled.x == 425
    assert scaled.y == 40
    assert scaled.width == 100
    assert scaled.height == 90
    assert scaled.weight == 1.5


def test_roi_manager_analyzes_regions_after_scaling() -> None:
    manager = ROIManager()
    manager.add_roi(ROI(name="door", x=100, y=20, width=50, height=20))

    regions = [MotionRegion(bbox=(55, 12, 6, 6), area=36.0)]
    hits = manager.analyze_regions(
        regions,
        frame_shape=(50, 100),
        source_shape=(100, 200),
    )

    assert len(hits) == 1
    assert hits[0].roi_name == "door"
    assert hits[0].region_count == 1
