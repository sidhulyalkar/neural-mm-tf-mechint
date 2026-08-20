from neural_mechint.circuits import area_under_faithfulness_curve, rank_components


def test_rank_components_uses_absolute_importance():
    ranked = rank_components(["a", "b", "c"], [0.2, -0.9, 0.4])
    assert ranked[0].name == "b"


def test_faithfulness_auc():
    auc = area_under_faithfulness_curve([0.0, 0.5, 1.0], [0.0, 1.0, 1.0])
    assert abs(auc - 0.75) < 1e-8
