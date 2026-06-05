import os

import pytest


def test_main_window_can_be_created_offscreen():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication

    from tmm_device_sim.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.layer_table.rowCount() >= 1
    assert "Unit" in [window.layer_table.horizontalHeaderItem(i).text() for i in range(window.layer_table.columnCount())]
    assert "Data" in [window.layer_table.horizontalHeaderItem(i).text() for i in range(window.layer_table.columnCount())]
    assert window.windowTitle()
    window.close()


def test_main_window_can_compare_top_and_bottom_incidence():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication

    from tmm_device_sim.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    incidence_options = [window.incidence_combo.itemText(i) for i in range(window.incidence_combo.count())]
    assert "compare" in incidence_options

    window.incidence_combo.setCurrentText("compare")
    results = window._simulate_incidence_compare("raise")

    assert set(results) == {"top", "bottom"}
    assert results["top"].incidence == "top"
    assert results["bottom"].incidence == "bottom"
    assert len(results["top"].wavelength_nm) == len(results["bottom"].wavelength_nm)
    window.close()


def test_batch_panel_has_trace_wavelength_control_and_profile():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication

    from tmm_device_sim.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.wavelength_start.setValue(500.0)
    window.wavelength_stop.setValue(700.0)
    window.wavelength_step.setValue(100.0)
    window.sweep_start.setValue(20.0)
    window.sweep_stop.setValue(40.0)
    window.sweep_step.setValue(10.0)
    window.batch_trace_wavelength.setValue(600.0)

    result = window._simulate_batch("raise")
    trace = window._batch_profile(result)
    window._plot_batch(result)
    titles = [axis.get_title() for axis in window.batch_panel.figure.axes]

    assert window.batch_trace_wavelength.value() == 600.0
    assert trace.shape == result.thicknesses_nm.shape
    assert "600 nm" in titles
    window.close()


def test_trace_wavelength_enter_refreshes_cached_batch_without_resimulating(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")

    from PySide6.QtWidgets import QApplication

    from tmm_device_sim.gui import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.wavelength_start.setValue(500.0)
    window.wavelength_stop.setValue(700.0)
    window.wavelength_step.setValue(100.0)
    window.sweep_start.setValue(20.0)
    window.sweep_stop.setValue(40.0)
    window.sweep_step.setValue(10.0)
    window.batch_trace_wavelength.setValue(500.0)

    result = window._simulate_batch("raise")
    window.current_batch = result
    window.current_batch_compare_results = None
    window._plot_batch(result)

    def fail_if_resimulated(*_args, **_kwargs):
        raise AssertionError("changing trace wavelength must not rerun batch simulation")

    monkeypatch.setattr(window, "_simulate_batch", fail_if_resimulated)
    window.batch_trace_wavelength.setValue(700.0)
    window.batch_trace_wavelength.editingFinished.emit()
    titles = [axis.get_title() for axis in window.batch_panel.figure.axes]

    assert "700 nm" in titles
    window.close()
