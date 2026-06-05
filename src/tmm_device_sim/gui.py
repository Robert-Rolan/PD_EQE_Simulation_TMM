from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from .batch import BatchResult, ThicknessSweep, eqe_at_wavelength_vs_thickness, run_thickness_sweep
from .examples import default_layers, default_media
from .exports import export_batch_csv, export_field_csv, export_spectrum_csv
from .materials import OpticalData, OpticalDataError, TabulatedOpticalData, builtin_optical_data, load_nk_file
from .model import Layer, Medium, wavelength_grid
from .simulation import SimulationResult, simulate_stack


def _load_qt():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    return {
        "QApplication": QApplication,
        "QCheckBox": QCheckBox,
        "QComboBox": QComboBox,
        "QDialog": QDialog,
        "QDialogButtonBox": QDialogButtonBox,
        "QDoubleSpinBox": QDoubleSpinBox,
        "QFileDialog": QFileDialog,
        "QFormLayout": QFormLayout,
        "QGridLayout": QGridLayout,
        "QGroupBox": QGroupBox,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QMainWindow": QMainWindow,
        "QMessageBox": QMessageBox,
        "QPushButton": QPushButton,
        "QSizePolicy": QSizePolicy,
        "QSplitter": QSplitter,
        "QTabWidget": QTabWidget,
        "QTableWidget": QTableWidget,
        "QTableWidgetItem": QTableWidgetItem,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "Qt": Qt,
        "Figure": Figure,
        "FigureCanvas": FigureCanvas,
    }


qt = _load_qt()
QApplication = qt["QApplication"]
QCheckBox = qt["QCheckBox"]
QComboBox = qt["QComboBox"]
QDialog = qt["QDialog"]
QDialogButtonBox = qt["QDialogButtonBox"]
QDoubleSpinBox = qt["QDoubleSpinBox"]
QFileDialog = qt["QFileDialog"]
QFormLayout = qt["QFormLayout"]
QGridLayout = qt["QGridLayout"]
QGroupBox = qt["QGroupBox"]
QHBoxLayout = qt["QHBoxLayout"]
QLabel = qt["QLabel"]
QMainWindow = qt["QMainWindow"]
QMessageBox = qt["QMessageBox"]
QPushButton = qt["QPushButton"]
QSizePolicy = qt["QSizePolicy"]
QSplitter = qt["QSplitter"]
QTabWidget = qt["QTabWidget"]
QTableWidget = qt["QTableWidget"]
QTableWidgetItem = qt["QTableWidgetItem"]
QVBoxLayout = qt["QVBoxLayout"]
QWidget = qt["QWidget"]
Qt = qt["Qt"]
Figure = qt["Figure"]
FigureCanvas = qt["FigureCanvas"]


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5.5, 4.0), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)


class MainWindow(QMainWindow):
    layer_columns = ["Active", "Name", "Thickness nm", "Color", "Material", "Unit", "Data"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TMM Photodiode Simulator")
        self.resize(1280, 820)
        self.materials: dict[str, OpticalData] = {}
        self.layers: list[Layer] = default_layers()
        for layer in self.layers:
            self.materials[layer.name] = layer.optical_data
        self.bottom_medium, self.top_medium = default_media()
        self.current_result: SimulationResult | None = None
        self.current_compare_results: dict[str, SimulationResult] | None = None
        self.current_batch: BatchResult | None = None
        self.current_batch_compare_results: dict[str, BatchResult] | None = None
        self._updating_table = False

        self._build_ui()
        self._populate_layer_table()
        self._refresh_sweep_layers()
        self._draw_structure()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_plots())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.layer_table = QTableWidget(0, len(self.layer_columns))
        self.layer_table.setHorizontalHeaderLabels(self.layer_columns)
        self.layer_table.itemChanged.connect(self._on_table_changed)
        self.layer_table.setMinimumWidth(570)
        layout.addWidget(QLabel("Layer stack (bottom to top)"))
        layout.addWidget(self.layer_table, stretch=1)

        button_row = QHBoxLayout()
        add_button = QPushButton("Add Layer")
        remove_button = QPushButton("Remove Layer")
        import_button = QPushButton("Import n,k")
        add_button.clicked.connect(self._add_layer)
        remove_button.clicked.connect(self._remove_selected_layer)
        import_button.clicked.connect(self._import_material)
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        button_row.addWidget(import_button)
        layout.addLayout(button_row)

        layout.addWidget(self._build_simulation_group())
        layout.addWidget(self._build_batch_group())

        run_row = QHBoxLayout()
        run_single = QPushButton("Run Single")
        run_batch = QPushButton("Run Batch")
        export_data = QPushButton("Export CSV")
        export_png = QPushButton("Export PNG")
        run_single.clicked.connect(self._run_single)
        run_batch.clicked.connect(self._run_batch)
        export_data.clicked.connect(self._export_csv)
        export_png.clicked.connect(self._export_png)
        run_row.addWidget(run_single)
        run_row.addWidget(run_batch)
        run_row.addWidget(export_data)
        run_row.addWidget(export_png)
        layout.addLayout(run_row)
        return panel

    def _build_simulation_group(self) -> QGroupBox:
        group = QGroupBox("Simulation")
        form = QFormLayout(group)

        self.incidence_combo = QComboBox()
        self.incidence_combo.addItems(["top", "bottom", "compare"])
        self.incidence_combo.currentTextChanged.connect(lambda _: self._draw_structure())
        form.addRow("Incidence", self.incidence_combo)

        self.bottom_medium_combo = self._medium_combo("glass")
        self.top_medium_combo = self._medium_combo("air")
        form.addRow("Bottom medium", self.bottom_medium_combo)
        form.addRow("Top medium", self.top_medium_combo)

        self.wavelength_start = self._spin(300.0, 1.0, 10000.0, 1.0)
        self.wavelength_stop = self._spin(1200.0, 1.0, 10000.0, 1.0)
        self.wavelength_step = self._spin(5.0, 0.1, 1000.0, 0.5)
        form.addRow("lambda start nm", self.wavelength_start)
        form.addRow("lambda stop nm", self.wavelength_stop)
        form.addRow("lambda step nm", self.wavelength_step)

        self.field_scope_combo = QComboBox()
        self.field_scope_combo.addItems(["active", "stack", "custom"])
        form.addRow("Field depth", self.field_scope_combo)
        self.custom_depth_start = self._spin(0.0, 0.0, 100000.0, 1.0)
        self.custom_depth_stop = self._spin(300.0, 0.0, 100000.0, 1.0)
        self.depth_step = self._spin(2.0, 0.1, 1000.0, 0.5)
        form.addRow("Custom start nm", self.custom_depth_start)
        form.addRow("Custom stop nm", self.custom_depth_stop)
        form.addRow("Depth step nm", self.depth_step)
        return group

    def _build_batch_group(self) -> QGroupBox:
        group = QGroupBox("Batch thickness sweep")
        grid = QGridLayout(group)
        self.sweep_layer_combo = QComboBox()
        self.sweep_start = self._spin(20.0, 0.1, 100000.0, 1.0)
        self.sweep_stop = self._spin(120.0, 0.1, 100000.0, 1.0)
        self.sweep_step = self._spin(10.0, 0.1, 100000.0, 1.0)
        self.batch_trace_wavelength = self._spin(1100.0, 1.0, 10000.0, 1.0)
        self.batch_trace_wavelength.editingFinished.connect(self._refresh_batch_profile_from_cache)
        grid.addWidget(QLabel("Layer"), 0, 0)
        grid.addWidget(self.sweep_layer_combo, 0, 1)
        grid.addWidget(QLabel("Start nm"), 1, 0)
        grid.addWidget(self.sweep_start, 1, 1)
        grid.addWidget(QLabel("Stop nm"), 2, 0)
        grid.addWidget(self.sweep_stop, 2, 1)
        grid.addWidget(QLabel("Step nm"), 3, 0)
        grid.addWidget(self.sweep_step, 3, 1)
        grid.addWidget(QLabel("Trace lambda nm"), 4, 0)
        grid.addWidget(self.batch_trace_wavelength, 4, 1)
        return group

    def _build_plots(self) -> QWidget:
        self.tabs = QTabWidget()
        self.structure_panel = PlotPanel()
        self.eqe_panel = PlotPanel()
        self.field_panel = PlotPanel()
        self.batch_panel = PlotPanel()
        self.tabs.addTab(self.structure_panel, "Structure")
        self.tabs.addTab(self.eqe_panel, "EQE")
        self.tabs.addTab(self.field_panel, "Field")
        self.tabs.addTab(self.batch_panel, "Batch")
        return self.tabs

    def _spin(self, value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _medium_combo(self, default: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(["air", "vacuum", "glass", "silicon", "SiO2"])
        combo.setCurrentText(default)
        return combo

    def _populate_layer_table(self) -> None:
        self._updating_table = True
        self.layer_table.setRowCount(len(self.layers))
        for row, layer in enumerate(self.layers):
            active_item = QTableWidgetItem("")
            active_item.setFlags(active_item.flags() | Qt.ItemIsUserCheckable)
            active_item.setCheckState(Qt.Checked if layer.active else Qt.Unchecked)
            self.layer_table.setItem(row, 0, active_item)
            self.layer_table.setItem(row, 1, QTableWidgetItem(layer.name))
            self.layer_table.setItem(row, 2, QTableWidgetItem(f"{layer.thickness_nm:g}"))
            self.layer_table.setItem(row, 3, QTableWidgetItem(layer.color))
            self.layer_table.setItem(row, 4, QTableWidgetItem(layer.name))
            self.layer_table.setCellWidget(row, 5, self._unit_combo())
            self.layer_table.setCellWidget(row, 6, self._data_button(row))
        self.layer_table.resizeColumnsToContents()
        self._updating_table = False

    def _unit_combo(self, unit: str = "nm") -> QComboBox:
        combo = QComboBox()
        combo.addItem("nm", "nm")
        combo.addItem("μm", "um")
        combo.setCurrentIndex(1 if unit == "um" else 0)
        combo.currentIndexChanged.connect(lambda *_: self._on_table_changed())
        return combo

    def _data_button(self, row: int) -> QPushButton:
        button = QPushButton("View")
        button.clicked.connect(lambda *_args, row=row: self._show_layer_data(row))
        return button

    def _on_table_changed(self) -> None:
        if self._updating_table:
            return
        try:
            self.layers = self._read_layers_from_table()
            self._refresh_sweep_layers()
            self._draw_structure()
        except Exception as exc:
            self.statusBar().showMessage(str(exc))

    def _read_layers_from_table(self) -> list[Layer]:
        layers: list[Layer] = []
        for row in range(self.layer_table.rowCount()):
            active = self.layer_table.item(row, 0).checkState() == Qt.Checked
            name = self._cell_text(row, 1, f"Layer {row + 1}")
            thickness = float(self._cell_text(row, 2, "100"))
            color = self._cell_text(row, 3, "#9fb6d8")
            material_name = self._cell_text(row, 4, name)
            optical_data = self._optical_data_for_row(material_name, self._row_unit(row))
            if optical_data is None:
                raise OpticalDataError(f"material '{material_name}' is not loaded")
            layers.append(Layer(name, thickness, optical_data, color, active))
        return layers

    def _row_unit(self, row: int) -> str:
        widget = self.layer_table.cellWidget(row, 5)
        if isinstance(widget, QComboBox):
            return str(widget.currentData() or "nm")
        return "nm"

    def _optical_data_for_row(self, material_name: str, wavelength_unit: str) -> OpticalData | None:
        optical_data = self.materials.get(material_name)
        if isinstance(optical_data, TabulatedOpticalData):
            return optical_data.with_wavelength_unit(wavelength_unit)
        return optical_data

    def _cell_text(self, row: int, column: int, default: str) -> str:
        item = self.layer_table.item(row, column)
        if item is None or not item.text().strip():
            return default
        return item.text().strip()

    def _add_layer(self) -> None:
        self.materials.setdefault("New Layer", builtin_optical_data("glass"))
        self.layers.append(Layer("New Layer", 100.0, self.materials["New Layer"], "#9fb6d8"))
        self._populate_layer_table()
        self._refresh_sweep_layers()
        self._draw_structure()

    def _remove_selected_layer(self) -> None:
        row = self.layer_table.currentRow()
        if row < 0 or len(self.layers) <= 1:
            return
        self.layers.pop(row)
        self._populate_layer_table()
        self._refresh_sweep_layers()
        self._draw_structure()

    def _import_material(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import n,k data",
            "",
            "Optical data (*.csv *.txt *.mdf);;All files (*)",
        )
        if not path:
            return
        try:
            row = self.layer_table.currentRow()
            unit = self._row_unit(row) if row >= 0 else "nm"
            data = load_nk_file(path, wavelength_unit=unit)
            name = Path(path).stem
            self.materials[name] = data
            if row >= 0:
                self.layer_table.setItem(row, 4, QTableWidgetItem(name))
            self.statusBar().showMessage(f"Imported material: {name}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    def _show_layer_data(self, row: int) -> None:
        try:
            if row < 0 or row >= self.layer_table.rowCount():
                return
            layer_name = self._cell_text(row, 1, f"Layer {row + 1}")
            material_name = self._cell_text(row, 4, layer_name)
            optical_data = self._optical_data_for_row(material_name, self._row_unit(row))
            if optical_data is None:
                raise OpticalDataError(f"material '{material_name}' is not loaded")
            if not isinstance(optical_data, TabulatedOpticalData):
                values = optical_data.complex_index([550.0])
                QMessageBox.information(
                    self,
                    "Material data",
                    f"{material_name}\nconstant n={values[0].real:g}, k={values[0].imag:g}",
                )
                return
            dialog = QDialog(self)
            dialog.setWindowTitle(f"{material_name} data matrix")
            layout = QVBoxLayout(dialog)
            layout.addWidget(
                QLabel(
                    f"{material_name}: source unit {optical_data.wavelength_unit}, "
                    f"{len(optical_data.wavelength_nm)} rows"
                )
            )
            table = QTableWidget(len(optical_data.wavelength_nm), 4)
            table.setHorizontalHeaderLabels(["source_wavelength", "wavelength_nm", "n", "k"])
            for index in range(len(optical_data.wavelength_nm)):
                table.setItem(index, 0, QTableWidgetItem(f"{optical_data.source_wavelength[index]:.8g}"))
                table.setItem(index, 1, QTableWidgetItem(f"{optical_data.wavelength_nm[index]:.8g}"))
                table.setItem(index, 2, QTableWidgetItem(f"{optical_data.n[index]:.8g}"))
                table.setItem(index, 3, QTableWidgetItem(f"{optical_data.k[index]:.8g}"))
            table.resizeColumnsToContents()
            layout.addWidget(table)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.resize(620, 420)
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Material data", str(exc))

    def _refresh_sweep_layers(self) -> None:
        current_index = self.sweep_layer_combo.currentIndex() if hasattr(self, "sweep_layer_combo") else 0
        self.sweep_layer_combo.blockSignals(True)
        self.sweep_layer_combo.clear()
        for index, layer in enumerate(self.layers):
            self.sweep_layer_combo.addItem(f"{index}: {layer.name}", index)
        self.sweep_layer_combo.setCurrentIndex(max(0, min(current_index, self.sweep_layer_combo.count() - 1)))
        self.sweep_layer_combo.blockSignals(False)

    def _current_wavelengths(self) -> np.ndarray:
        return wavelength_grid(
            self.wavelength_start.value(),
            self.wavelength_stop.value(),
            self.wavelength_step.value(),
        )

    def _medium_from_combo(self, combo: QComboBox) -> Medium:
        name = combo.currentText().strip()
        optical_data = self.materials.get(name)
        if optical_data is None:
            optical_data = builtin_optical_data(name)
        return Medium(name, optical_data)

    def _simulation_kwargs(self, wavelength_bounds_policy: str = "raise", incidence: str | None = None) -> dict:
        field_scope = self.field_scope_combo.currentText()
        custom_range = None
        if field_scope == "custom":
            custom_range = (self.custom_depth_start.value(), self.custom_depth_stop.value())
        return {
            "bottom_medium": self._medium_from_combo(self.bottom_medium_combo),
            "top_medium": self._medium_from_combo(self.top_medium_combo),
            "wavelengths_nm": self._current_wavelengths(),
            "incidence": incidence or self.incidence_combo.currentText(),
            "field_scope": field_scope,
            "custom_depth_range_nm": custom_range,
            "depth_step_nm": self.depth_step.value(),
            "wavelength_bounds_policy": wavelength_bounds_policy,
        }

    def _run_single(self) -> None:
        try:
            self._run_single_with_policy("raise")
        except OpticalDataError as exc:
            if not self._is_wavelength_range_error(exc):
                QMessageBox.critical(self, "Simulation failed", str(exc))
                return
            policy = self._ask_wavelength_policy(exc)
            if policy is None:
                return
            try:
                self._run_single_with_policy(policy)
            except Exception as retry_exc:
                QMessageBox.critical(self, "Simulation failed", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Simulation failed", str(exc))
            return
        if self.current_compare_results is not None:
            self._plot_eqe_compare(self.current_compare_results)
            self._plot_field(self.current_compare_results["top"])
        elif self.current_result is not None:
            self._plot_eqe(self.current_result)
            self._plot_field(self.current_result)
        self.tabs.setCurrentWidget(self.eqe_panel)
        self.statusBar().showMessage("Single simulation complete")

    def _run_batch(self) -> None:
        try:
            self._run_batch_with_policy("raise")
        except OpticalDataError as exc:
            if not self._is_wavelength_range_error(exc):
                QMessageBox.critical(self, "Batch simulation failed", str(exc))
                return
            policy = self._ask_wavelength_policy(exc)
            if policy is None:
                return
            try:
                self._run_batch_with_policy(policy)
            except Exception as retry_exc:
                QMessageBox.critical(self, "Batch simulation failed", str(retry_exc))
                return
        except Exception as exc:
            QMessageBox.critical(self, "Batch simulation failed", str(exc))
            return
        if self.current_batch_compare_results is not None:
            self._plot_batch_compare(self.current_batch_compare_results)
        elif self.current_batch is not None:
            self._plot_batch(self.current_batch)
        self.tabs.setCurrentWidget(self.batch_panel)
        self.statusBar().showMessage("Batch simulation complete")

    def _run_single_with_policy(self, wavelength_bounds_policy: str) -> None:
        self.current_compare_results = None
        self.current_result = None
        if self.incidence_combo.currentText() == "compare":
            self.current_compare_results = self._simulate_incidence_compare(wavelength_bounds_policy)
            self.current_result = self.current_compare_results["top"]
        else:
            self.current_result = self._simulate_single(wavelength_bounds_policy)

    def _run_batch_with_policy(self, wavelength_bounds_policy: str) -> None:
        self.current_batch_compare_results = None
        self.current_batch = None
        if self.incidence_combo.currentText() == "compare":
            self.current_batch_compare_results = self._simulate_batch_incidence_compare(wavelength_bounds_policy)
            self.current_batch = self.current_batch_compare_results["top"]
        else:
            self.current_batch = self._simulate_batch(wavelength_bounds_policy)

    def _simulate_single(self, wavelength_bounds_policy: str) -> SimulationResult:
        self.layers = self._read_layers_from_table()
        return simulate_stack(self.layers, **self._simulation_kwargs(wavelength_bounds_policy))

    def _simulate_incidence_compare(self, wavelength_bounds_policy: str) -> dict[str, SimulationResult]:
        self.layers = self._read_layers_from_table()
        return {
            "top": simulate_stack(self.layers, **self._simulation_kwargs(wavelength_bounds_policy, incidence="top")),
            "bottom": simulate_stack(self.layers, **self._simulation_kwargs(wavelength_bounds_policy, incidence="bottom")),
        }

    def _simulate_batch(self, wavelength_bounds_policy: str) -> BatchResult:
        self.layers = self._read_layers_from_table()
        layer_index = int(self.sweep_layer_combo.currentData())
        sweep = ThicknessSweep(
            layer_index=layer_index,
            start_nm=self.sweep_start.value(),
            stop_nm=self.sweep_stop.value(),
            step_nm=self.sweep_step.value(),
        )
        kwargs = self._simulation_kwargs(wavelength_bounds_policy)
        return run_thickness_sweep(
            self.layers,
            bottom_medium=kwargs["bottom_medium"],
            top_medium=kwargs["top_medium"],
            wavelengths_nm=kwargs["wavelengths_nm"],
            incidence=kwargs["incidence"],
            sweep=sweep,
            wavelength_bounds_policy=wavelength_bounds_policy,
        )

    def _simulate_batch_incidence_compare(self, wavelength_bounds_policy: str) -> dict[str, BatchResult]:
        self.layers = self._read_layers_from_table()
        layer_index = int(self.sweep_layer_combo.currentData())
        sweep = ThicknessSweep(
            layer_index=layer_index,
            start_nm=self.sweep_start.value(),
            stop_nm=self.sweep_stop.value(),
            step_nm=self.sweep_step.value(),
        )
        top_kwargs = self._simulation_kwargs(wavelength_bounds_policy, incidence="top")
        bottom_kwargs = self._simulation_kwargs(wavelength_bounds_policy, incidence="bottom")
        return {
            "top": run_thickness_sweep(
                self.layers,
                bottom_medium=top_kwargs["bottom_medium"],
                top_medium=top_kwargs["top_medium"],
                wavelengths_nm=top_kwargs["wavelengths_nm"],
                incidence="top",
                sweep=sweep,
                wavelength_bounds_policy=wavelength_bounds_policy,
            ),
            "bottom": run_thickness_sweep(
                self.layers,
                bottom_medium=bottom_kwargs["bottom_medium"],
                top_medium=bottom_kwargs["top_medium"],
                wavelengths_nm=bottom_kwargs["wavelengths_nm"],
                incidence="bottom",
                sweep=sweep,
                wavelength_bounds_policy=wavelength_bounds_policy,
            ),
        }

    def _is_wavelength_range_error(self, exc: OpticalDataError) -> bool:
        text = str(exc).lower()
        return "outside" in text or "wavelength range" in text or "no requested wavelengths" in text

    def _ask_wavelength_policy(self, exc: OpticalDataError) -> str | None:
        box = QMessageBox(self)
        box.setWindowTitle("n,k wavelength range")
        box.setText("Imported n,k data do not cover the full simulation wavelength range.")
        box.setInformativeText(str(exc))
        clip_button = box.addButton("Fit covered wavelengths only", QMessageBox.ButtonRole.AcceptRole)
        extrapolate_button = box.addButton("Use extrapolated data", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked == clip_button:
            return "clip"
        if clicked == extrapolate_button:
            return "extrapolate"
        return None

    def _draw_structure(self) -> None:
        figure = self.structure_panel.figure
        figure.clear()
        axis = figure.add_subplot(111)
        layers = self.layers
        for index, layer in enumerate(layers):
            axis.add_patch(
                axis.barh(index, 1.0, left=0.0, height=0.92, color=layer.color, edgecolor="white")[0]
            )
            axis.text(
                0.5,
                index,
                f"{layer.name}\n{layer.thickness_nm:g} nm",
                ha="center",
                va="center",
                fontsize=9,
            )
        top_y = len(layers) - 0.5
        bottom_y = -0.5
        incidence = self.incidence_combo.currentText()
        if incidence in {"top", "compare"}:
            axis.annotate("light", xy=(1.15, top_y - 0.6), xytext=(1.15, top_y + 0.8), arrowprops={"arrowstyle": "->"})
        if incidence in {"bottom", "compare"}:
            axis.annotate("light", xy=(1.15, bottom_y + 0.6), xytext=(1.15, bottom_y - 0.8), arrowprops={"arrowstyle": "->"})
        axis.set_xlim(-0.05, 1.35)
        axis.set_ylim(-0.8, len(layers) - 0.2)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title("Device structure (bottom to top)")
        for spine in axis.spines.values():
            spine.set_visible(False)
        self.structure_panel.canvas.draw_idle()

    def _plot_eqe(self, result: SimulationResult) -> None:
        figure = self.eqe_panel.figure
        figure.clear()
        axis = figure.add_subplot(111)
        axis.plot(result.wavelength_nm, result.eqe * 100.0, color="#008b8b", linewidth=2.0, label="Ideal EQE")
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("EQE (%)")
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=0.25)
        axis.legend()
        self.eqe_panel.canvas.draw_idle()

    def _plot_eqe_compare(self, results: dict[str, SimulationResult]) -> None:
        figure = self.eqe_panel.figure
        figure.clear()
        axis = figure.add_subplot(111)
        axis.plot(
            results["top"].wavelength_nm,
            results["top"].eqe * 100.0,
            color="#008b8b",
            linewidth=2.0,
            label="Top incidence",
        )
        axis.plot(
            results["bottom"].wavelength_nm,
            results["bottom"].eqe * 100.0,
            color="#c45a2a",
            linewidth=2.0,
            label="Bottom incidence",
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("EQE (%)")
        axis.set_ylim(bottom=0)
        axis.grid(True, alpha=0.25)
        axis.legend()
        self.eqe_panel.canvas.draw_idle()

    def _plot_field(self, result: SimulationResult) -> None:
        figure = self.field_panel.figure
        figure.clear()
        axis = figure.add_subplot(111)
        image = axis.imshow(
            result.field_intensity,
            origin="lower",
            aspect="auto",
            extent=[
                result.wavelength_nm[0],
                result.wavelength_nm[-1],
                result.field_depth_nm[0],
                result.field_depth_nm[-1],
            ],
            cmap="jet",
        )
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("Distance from bottom (nm)")
        axis.set_title("|E/E0|^2")
        figure.colorbar(image, ax=axis)
        self.field_panel.canvas.draw_idle()

    def _plot_batch(self, result: BatchResult) -> None:
        figure = self.batch_panel.figure
        figure.clear()
        grid = figure.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
        eqe_axis = figure.add_subplot(grid[0, 0])
        map_axis = figure.add_subplot(grid[0, 1])
        profile_axis = figure.add_subplot(grid[0, 2])
        for index, thickness in enumerate(result.thicknesses_nm):
            eqe_axis.plot(result.wavelength_nm, result.eqe_map[index] * 100.0, linewidth=1.2, label=f"{thickness:g} nm")
        eqe_axis.set_xlabel("Wavelength (nm)")
        eqe_axis.set_ylabel("EQE (%)")
        eqe_axis.grid(True, alpha=0.25)
        if len(result.thicknesses_nm) <= 12:
            eqe_axis.legend(fontsize=8)

        image = map_axis.imshow(
            result.eqe_map * 100.0,
            origin="lower",
            aspect="auto",
            extent=[
                result.wavelength_nm[0],
                result.wavelength_nm[-1],
                result.thicknesses_nm[0],
                result.thicknesses_nm[-1],
            ],
            cmap="viridis",
        )
        map_axis.set_xlabel("Wavelength (nm)")
        map_axis.set_ylabel("Swept thickness (nm)")
        map_axis.set_title("Ideal EQE (%)")
        figure.colorbar(image, ax=map_axis)

        trace = self._batch_profile(result)
        trace_wavelength = self.batch_trace_wavelength.value()
        profile_axis.plot(result.thicknesses_nm, trace * 100.0, color="#444444", marker="o", linewidth=1.6, markersize=3)
        profile_axis.set_xlabel("Swept thickness (nm)")
        profile_axis.set_ylabel("EQE (%)")
        profile_axis.set_title(f"{trace_wavelength:g} nm")
        profile_axis.grid(True, alpha=0.25)
        self.batch_panel.canvas.draw_idle()

    def _plot_batch_compare(self, results: dict[str, BatchResult]) -> None:
        figure = self.batch_panel.figure
        figure.clear()
        top_result = results["top"]
        bottom_result = results["bottom"]
        grid = figure.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
        top_axis = figure.add_subplot(grid[0, 0])
        bottom_axis = figure.add_subplot(grid[0, 1])
        profile_axis = figure.add_subplot(grid[0, 2])
        vmax = max(float(np.max(top_result.eqe_map)), float(np.max(bottom_result.eqe_map))) * 100.0
        for axis, result, title in [
            (top_axis, top_result, "Top incidence EQE (%)"),
            (bottom_axis, bottom_result, "Bottom incidence EQE (%)"),
        ]:
            image = axis.imshow(
                result.eqe_map * 100.0,
                origin="lower",
                aspect="auto",
                extent=[
                    result.wavelength_nm[0],
                    result.wavelength_nm[-1],
                    result.thicknesses_nm[0],
                    result.thicknesses_nm[-1],
                ],
                cmap="viridis",
                vmin=0.0,
                vmax=vmax if vmax > 0 else None,
            )
            axis.set_xlabel("Wavelength (nm)")
            axis.set_ylabel("Swept thickness (nm)")
            axis.set_title(title)
            figure.colorbar(image, ax=axis)

        trace_wavelength = self.batch_trace_wavelength.value()
        profile_axis.plot(
            top_result.thicknesses_nm,
            self._batch_profile(top_result) * 100.0,
            color="#008b8b",
            marker="o",
            linewidth=1.6,
            markersize=3,
            label="Top",
        )
        profile_axis.plot(
            bottom_result.thicknesses_nm,
            self._batch_profile(bottom_result) * 100.0,
            color="#c45a2a",
            marker="s",
            linewidth=1.6,
            markersize=3,
            label="Bottom",
        )
        profile_axis.set_xlabel("Swept thickness (nm)")
        profile_axis.set_ylabel("EQE (%)")
        profile_axis.set_title(f"{trace_wavelength:g} nm")
        profile_axis.grid(True, alpha=0.25)
        profile_axis.legend()
        self.batch_panel.canvas.draw_idle()

    def _batch_profile(self, result: BatchResult) -> np.ndarray:
        return eqe_at_wavelength_vs_thickness(result, self.batch_trace_wavelength.value())

    def _refresh_batch_profile_from_cache(self) -> None:
        try:
            if self.current_batch_compare_results is not None:
                self._plot_batch_compare(self.current_batch_compare_results)
            elif self.current_batch is not None:
                self._plot_batch(self.current_batch)
        except Exception as exc:
            self.statusBar().showMessage(str(exc))

    def _export_csv(self) -> None:
        try:
            if self.tabs.currentWidget() == self.batch_panel and self.current_batch is not None:
                path, _ = QFileDialog.getSaveFileName(self, "Export batch EQE CSV", "batch_eqe.csv", "CSV (*.csv)")
                if path:
                    export_batch_csv(self.current_batch, path)
                    self.statusBar().showMessage(f"Exported {path}")
                return
            if self.current_result is None:
                raise RuntimeError("run a simulation before exporting data")
            path, _ = QFileDialog.getSaveFileName(self, "Export spectrum CSV", "spectrum.csv", "CSV (*.csv)")
            if not path:
                return
            export_spectrum_csv(self.current_result, path)
            field_path = Path(path).with_name(f"{Path(path).stem}_field.csv")
            export_field_csv(self.current_result, field_path)
            self.statusBar().showMessage(f"Exported {path} and {field_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _export_png(self) -> None:
        figure = self._current_figure()
        path, _ = QFileDialog.getSaveFileName(self, "Export PNG", "figure.png", "PNG (*.png)")
        if not path:
            return
        try:
            figure.savefig(path, dpi=220)
            self.statusBar().showMessage(f"Exported {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def _current_figure(self):
        widget = self.tabs.currentWidget()
        if widget == self.eqe_panel:
            return self.eqe_panel.figure
        if widget == self.field_panel:
            return self.field_panel.figure
        if widget == self.batch_panel:
            return self.batch_panel.figure
        return self.structure_panel.figure


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()
