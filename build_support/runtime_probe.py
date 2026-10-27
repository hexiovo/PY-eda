"""独立运行时/本机DLL冻结探针，不作为完整软件或干净系统验收。"""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import numpy as np
from scipy import signal
from statsmodels.regression.mixed_linear_model import MixedLM
from PySide6 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

if __name__ == '__main__':
    application = QtWidgets.QApplication([])
    widget = QtWidgets.QLabel('EDA 中文运行时探针')
    widget.resize(320, 100)
    widget.show()
    application.processEvents()
    sos = signal.butter(2, .2, output='sos')
    filtered = signal.sosfiltfilt(sos, np.sin(np.arange(100)))
    figure = Figure(figsize=(3, 2))
    figure.add_subplot().plot(filtered)
    FigureCanvasAgg(figure).draw()
    print(json.dumps({'qt': QtCore.qVersion(), 'numpy': np.__version__, 'filter_finite': bool(np.isfinite(filtered).all()),
                      'mixedlm_import': callable(MixedLM), 'qt_widget_valid': not widget.grab().isNull(),
                      'scope': 'development_machine_runtime_probe_not_clean_machine_test'}))
    widget.close()
