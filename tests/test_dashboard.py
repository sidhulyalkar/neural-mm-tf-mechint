from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard.py"


def test_missing_run_has_actionable_empty_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    assert not app.exception
    assert "python train.py" in app.code[0].value


def test_saved_run_attention_and_head_intervention(trained_run, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(DASHBOARD)).run(timeout=20)
    app.sidebar.text_input[0].set_value(str(trained_run)).run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 3
    app.multiselect[0].set_value([0]).run(timeout=20)
    app.button[0].click().run(timeout=20)
    assert not app.exception
    assert any("ablated MSE" in entry.value for entry in app.markdown)
