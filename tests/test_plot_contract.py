import numpy as np
import pandas as pd
from eda_toolkit import core_plot


def test_quality_figure_contains_raw_processed_events_and_scr(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(core_plot, "_finish_figure", lambda figure, *args, **kwargs: captured.append(figure) or "figure.png")
    time = np.arange(5.)
    bundle = {"time_s": time, "clean_uS": np.array([1., 1., 2., 1., 1.]),
              "metadata": {"recording_id": "r"}, "quality": {"eligible": np.ones(5,dtype=bool)},
              "record": {"time_s": time, "eda_uS": np.array([1., 3., 2., 4., 1.])}}
    result = {"bundle": bundle, "tables": {"scr": pd.DataFrame([{"peak_s":2.,"amplitude_uS":1.}]),
              "windows": pd.DataFrame([{"event_onset_s":1.5}])}}
    core_plot._record_figure(result,tmp_path,core_plot._font_properties()[0])
    axis = captured[0].axes[0]
    labels = {line.get_label():line for line in axis.lines}
    np.testing.assert_array_equal(labels["原始电导 · μS"].get_ydata(), [1,3,2,4,1])
    np.testing.assert_array_equal(labels["清理电导 · μS"].get_ydata(), [1,1,2,1,1])
    np.testing.assert_array_equal(labels["事件"].get_xdata(), [1.5,1.5])
    assert any(artist.get_label()=="SCR 峰" for artist in axis.collections)
