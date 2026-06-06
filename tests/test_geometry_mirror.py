"""Pin contract: system1/geometry.py must produce identical bearings to the
canonical implementation in drone-detection-ml.

These two files were copy-forked because S1 ships in a slim container without
the full ML package. They MUST stay identical — silent drift produces wrong
GPS positions downstream with no error.

Skips if drone-detection-ml isn't on the path (CI installs it; the prod
image does not).
"""

from __future__ import annotations

import math

import pytest

from system1.geometry import bbox_center_to_bearing as s1_bearing

try:
    from drone_detector.geometry import bbox_center_to_bearing as ml_bearing
except ImportError:
    pytest.skip("drone-detection-ml not installed (install with `pip install -e ../drone-detection-ml`)",
                allow_module_level=True)


_CASES = [
    # (u, v, w, h, az, el, hfov, vfov)
    (320, 240, 640, 480, 0.0,   0.0,  90.0, 60.0),  # center
    (640, 240, 640, 480, 0.0,   0.0,  90.0, 60.0),  # right edge
    (320,   0, 640, 480, 0.0,   0.0,  90.0, 60.0),  # top edge → +U
    (160, 360, 640, 480, 45.0,  10.0, 70.0, 50.0),
    (480, 120, 640, 480, -90.0, -5.0, 60.0, 40.0),
]


@pytest.mark.parametrize("case", _CASES)
def test_bearing_matches_ml_package(case):
    a = s1_bearing(*case)
    b = ml_bearing(*case)
    for x, y in zip(a, b):
        assert math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12), (case, a, b)
