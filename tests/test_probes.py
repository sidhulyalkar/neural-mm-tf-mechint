import numpy as np

from neural_mechint.probes import mean_difference_direction, probe_with_controls


def test_mean_difference_direction_unit_norm():
    positive = np.array([[2.0, 0.0], [3.0, 0.0]])
    negative = np.array([[0.0, 0.0], [1.0, 0.0]])
    direction = mean_difference_direction(positive, negative)
    np.testing.assert_allclose(direction, [1.0, 0.0])


def test_probe_detects_strong_signal():
    rng = np.random.default_rng(3)
    y = np.array([0] * 30 + [1] * 30)
    x = rng.normal(size=(60, 8))
    x[:, 0] += y * 4.0
    result = probe_with_controls(x, y, folds=5, seed=1)
    assert result.roc_auc > 0.9
    assert result.shuffled_roc_auc < 0.8
