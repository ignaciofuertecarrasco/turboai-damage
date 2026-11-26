"""Geometry of the deviation input and the published numbers of the README."""
import json

import numpy as np
import pytest

from turboai_damage import RESULTS


def test_bump_on_suction_side_gives_local_outward_deviation():
    from turboai_damage.deviation_model import XS, deviation
    d = deviation({}, [{"side": "lower", "s0": 0.6, "w": 0.05, "amp": 0.006}])
    assert d.shape == (2, len(XS))
    assert d[0].max() > 0.004                              # outward on the suction side, close to the amplitude
    assert np.abs(d[1]).max() < 1e-3                       # pressure side untouched
    assert np.abs(d[0][:20]).max() < 1e-3                  # local: nothing near the leading edge
    assert np.allclose(deviation({"a2": 0.01}), 0.0, atol=1e-9)   # a family member has no deviation


@pytest.mark.skipif(not (RESULTS / "hybrid" / "metrics.json").exists(), reason="results not present")
def test_readme_numbers():
    m = json.loads((RESULTS / "hybrid" / "metrics.json").read_text())
    assert m["defects"]["n"] == 48
    assert m["defects"]["dzeta_correlation"] == pytest.approx(0.70, abs=0.01)
    assert m["blind"]["dzeta_correlation"] == pytest.approx(0.78, abs=0.01)
    ood = m["ood"]
    assert 1 - ood["hybrid_wall_mis_mae"] / ood["parametric_wall_mis_mae"] == pytest.approx(0.41, abs=0.01)
    d = m["defects"]
    assert d["dmis_parametric_error_region"] / d["dmis_hybrid_error_region"] > 4.0
