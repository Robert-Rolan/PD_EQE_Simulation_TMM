import numpy as np

from tmm_device_sim.batch import ThicknessSweep, eqe_at_wavelength_vs_thickness, run_thickness_sweep
from tmm_device_sim.materials import ConstantOpticalData
from tmm_device_sim.model import Layer, Medium


def test_single_layer_thickness_sweep_preserves_original_layers():
    layers = [
        Layer("Transport", 30.0, ConstantOpticalData(1.8, 0.0), "#a7d8f0"),
        Layer("Active", 100.0, ConstantOpticalData(2.0, 0.1), "#8a99b6", active=True),
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))
    sweep = ThicknessSweep(layer_index=0, start_nm=20.0, stop_nm=40.0, step_nm=10.0)

    result = run_thickness_sweep(
        layers,
        bottom_medium=air,
        top_medium=air,
        wavelengths_nm=np.array([500.0, 600.0, 700.0]),
        sweep=sweep,
    )

    assert np.allclose(result.thicknesses_nm, [20.0, 30.0, 40.0])
    assert result.eqe_map.shape == (3, 3)
    assert layers[0].thickness_nm == 30.0


def test_thickness_sweep_forwards_wavelength_clip_policy():
    from tmm_device_sim.materials import TabulatedOpticalData

    layers = [
        Layer(
            "Active",
            100.0,
            TabulatedOpticalData(
                wavelength_nm=np.array([500.0, 700.0]),
                n=np.array([2.0, 2.0]),
                k=np.array([0.1, 0.1]),
            ),
            "#8a99b6",
            active=True,
        )
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))

    result = run_thickness_sweep(
        layers,
        bottom_medium=air,
        top_medium=air,
        wavelengths_nm=np.array([400.0, 500.0, 600.0, 700.0, 800.0]),
        sweep=ThicknessSweep(layer_index=0, start_nm=80.0, stop_nm=100.0, step_nm=20.0),
        wavelength_bounds_policy="clip",
    )

    assert np.allclose(result.wavelength_nm, [500.0, 600.0, 700.0])
    assert result.eqe_map.shape == (2, 3)


def test_eqe_at_wavelength_vs_thickness_interpolates_target_wavelength():
    layers = [
        Layer("Active", 100.0, ConstantOpticalData(2.0, 0.1), "#8a99b6", active=True),
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))
    result = run_thickness_sweep(
        layers,
        bottom_medium=air,
        top_medium=air,
        wavelengths_nm=np.array([500.0, 600.0, 700.0]),
        sweep=ThicknessSweep(layer_index=0, start_nm=80.0, stop_nm=120.0, step_nm=20.0),
    )

    trace = eqe_at_wavelength_vs_thickness(result, 550.0)

    assert trace.shape == result.thicknesses_nm.shape
    assert np.allclose(trace, 0.5 * (result.eqe_map[:, 0] + result.eqe_map[:, 1]))
