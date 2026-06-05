from tmm_device_sim.exports import export_batch_csv, export_field_csv, export_spectrum_csv
from tmm_device_sim.batch import ThicknessSweep, run_thickness_sweep
from tmm_device_sim.materials import ConstantOpticalData
from tmm_device_sim.model import Layer, Medium
from tmm_device_sim.simulation import simulate_stack


def test_exports_single_simulation_csv_files(tmp_path):
    layers = [Layer("Active", 100.0, ConstantOpticalData(2.0, 0.1), "#8a99b6", active=True)]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))
    result = simulate_stack(layers, bottom_medium=air, top_medium=air, wavelengths_nm=[500.0, 600.0])

    spectrum_path = tmp_path / "spectrum.csv"
    field_path = tmp_path / "field.csv"
    export_spectrum_csv(result, spectrum_path)
    export_field_csv(result, field_path)

    assert spectrum_path.read_text(encoding="utf-8").splitlines()[0].startswith("wavelength_nm")
    assert "ideal_eqe" in spectrum_path.read_text(encoding="utf-8")
    assert field_path.read_text(encoding="utf-8").splitlines()[0].startswith("depth_nm")


def test_exports_batch_eqe_csv(tmp_path):
    layers = [Layer("Active", 100.0, ConstantOpticalData(2.0, 0.1), "#8a99b6", active=True)]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))
    batch = run_thickness_sweep(
        layers,
        bottom_medium=air,
        top_medium=air,
        wavelengths_nm=[500.0, 600.0],
        sweep=ThicknessSweep(0, 80.0, 100.0, 20.0),
    )

    path = tmp_path / "batch.csv"
    export_batch_csv(batch, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("thickness_nm")
    assert len(lines) == 3
