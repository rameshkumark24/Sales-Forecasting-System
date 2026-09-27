"""Smoke test: the dashboard renders against the committed models and data."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_errors():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == [
        "Overview",
        "Store explorer",
        "Model performance",
        "Forecast table",
    ]
    labels = [m.label for m in at.metric]
    assert "Total forecast revenue" in labels
    assert not at.info, "app fell back to retraining; committed models are unusable"


def test_filters_apply():
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.multiselect[0].select("Europe").run()
    assert not at.exception, at.exception
    stores = next(m for m in at.metric if m.label == "Stores")
    assert 0 < int(stores.value) < 185
