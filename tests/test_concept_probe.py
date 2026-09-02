import numpy as np
import pytest

pytest.importorskip("sklearn")
from interpretability.cav_analysis import compute_cav


def test_binary_concept_direction_and_multiclass_rejection():
    x = np.array([[-3.0, 0.0], [-2.0, 1.0], [2.0, 0.0], [3.0, 1.0]])
    direction = compute_cav(x, np.array([0, 0, 1, 1]))
    assert direction.shape == (2,)
    assert direction[0] > abs(direction[1])
    with pytest.raises(ValueError, match="both 0 and 1"):
        compute_cav(x, [0, 1, 2, 2])
