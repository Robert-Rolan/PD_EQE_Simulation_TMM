import numpy as np

from tmm_device_sim.materials import ConstantOpticalData
from tmm_device_sim.model import Layer, Medium
from tmm_device_sim.simulation import simulate_stack


def test_lossless_single_layer_conserves_energy():
    layers = [
        Layer(
            name="Dielectric",
            thickness_nm=120.0,
            optical_data=ConstantOpticalData(1.5, 0.0),
            color="#aaccee",
        )
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))

    result = simulate_stack(layers, bottom_medium=air, top_medium=air, wavelengths_nm=[500.0, 650.0])

    assert np.allclose(result.reflectance + result.transmittance + result.total_absorption, 1.0, atol=1e-8)
    assert np.allclose(result.total_absorption, 0.0, atol=1e-8)
    assert np.allclose(result.layer_absorption["Dielectric"], 0.0, atol=1e-8)


def test_eqe_counts_only_active_layer_absorption():
    layers = [
        Layer(
            name="Active BHJ",
            thickness_nm=200.0,
            optical_data=ConstantOpticalData(2.0, 0.15),
            color="#8394b5",
            active=True,
        ),
        Layer(
            name="Metal",
            thickness_nm=80.0,
            optical_data=ConstantOpticalData(0.2, 3.0),
            color="#808080",
            active=False,
        ),
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))

    result = simulate_stack(layers, bottom_medium=air, top_medium=air, wavelengths_nm=[600.0, 800.0])

    assert np.all(result.layer_absorption["Active BHJ"] > 0)
    assert np.all(result.layer_absorption["Metal"] > 0)
    assert np.allclose(result.eqe, result.layer_absorption["Active BHJ"])
    assert np.all(result.eqe < result.total_absorption)


def test_top_and_bottom_incidence_both_return_bottom_referenced_depths():
    layers = [
        Layer("Layer A", 50.0, ConstantOpticalData(1.6, 0.02), "#ccddee"),
        Layer("Layer B", 70.0, ConstantOpticalData(2.1, 0.03), "#ddccaa", active=True),
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))
    glass = Medium("glass", ConstantOpticalData(1.5, 0.0))

    top = simulate_stack(layers, bottom_medium=glass, top_medium=air, wavelengths_nm=[550.0], incidence="top")
    bottom = simulate_stack(layers, bottom_medium=glass, top_medium=air, wavelengths_nm=[550.0], incidence="bottom")

    assert np.all(np.diff(top.field_depth_nm) >= 0)
    assert np.all(np.diff(bottom.field_depth_nm) >= 0)
    assert top.field_intensity.shape == bottom.field_intensity.shape
    assert np.allclose(top.reflectance + top.transmittance + top.total_absorption, 1.0, atol=1e-6)
    assert np.allclose(bottom.reflectance + bottom.transmittance + bottom.total_absorption, 1.0, atol=1e-6)


def test_clip_policy_uses_only_wavelengths_covered_by_tabulated_layers():
    from tmm_device_sim.materials import TabulatedOpticalData

    layers = [
        Layer(
            name="Active",
            thickness_nm=100.0,
            optical_data=TabulatedOpticalData(
                wavelength_nm=np.array([500.0, 700.0]),
                n=np.array([2.0, 2.0]),
                k=np.array([0.1, 0.1]),
            ),
            color="#8a99b6",
            active=True,
        )
    ]
    air = Medium("air", ConstantOpticalData(1.0, 0.0))

    result = simulate_stack(
        layers,
        bottom_medium=air,
        top_medium=air,
        wavelengths_nm=np.array([400.0, 500.0, 600.0, 700.0, 800.0]),
        wavelength_bounds_policy="clip",
    )

    assert np.allclose(result.wavelength_nm, [500.0, 600.0, 700.0])
