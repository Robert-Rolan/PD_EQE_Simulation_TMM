import numpy as np
import pytest

from tmm_device_sim.materials import OpticalDataError, load_nk_file


def test_loads_csv_with_semantic_headers_and_interpolates(tmp_path):
    path = tmp_path / "bhj.csv"
    path.write_text(
        "wavelength_nm,refractive_index,extinction_coefficient\n"
        "300,1.50,0.10\n"
        "400,1.70,0.20\n"
        "500,1.90,0.30\n",
        encoding="utf-8",
    )

    data = load_nk_file(path)
    values = data.complex_index(np.array([350.0, 450.0]))

    assert np.allclose(values.real, [1.60, 1.80])
    assert np.allclose(values.imag, [0.15, 0.25])


def test_loads_space_delimited_txt_with_lambda_header(tmp_path):
    path = tmp_path / "zno.txt"
    path.write_text(
        "lambda n k\n"
        "300 2.0 0.0\n"
        "500 2.2 0.02\n",
        encoding="utf-8",
    )

    data = load_nk_file(path)

    assert np.allclose(data.complex_index([400.0]), [2.1 + 0.01j])


def test_missing_required_column_raises_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("wavelength,n\n300,1.5\n400,1.6\n", encoding="utf-8")

    with pytest.raises(OpticalDataError, match="k"):
        load_nk_file(path)


def test_interpolation_refuses_out_of_range_wavelengths(tmp_path):
    path = tmp_path / "ito.csv"
    path.write_text(
        "wavelength,n,k\n"
        "400,1.8,0.02\n"
        "800,1.7,0.01\n",
        encoding="utf-8",
    )

    data = load_nk_file(path)

    with pytest.raises(OpticalDataError, match="outside"):
        data.complex_index([350.0])


def test_load_can_force_wavelength_unit_to_um(tmp_path):
    path = tmp_path / "organic.csv"
    path.write_text(
        "wavelength,n,k\n"
        "0.4,1.8,0.1\n"
        "0.8,2.0,0.2\n",
        encoding="utf-8",
    )

    data = load_nk_file(path, wavelength_unit="um")

    assert np.allclose(data.wavelength_nm, [400.0, 800.0])
    assert np.allclose(data.complex_index([600.0]), [1.9 + 0.15j])


def test_mdf_file_with_metadata_header_is_supported(tmp_path):
    path = tmp_path / "transport.mdf"
    path.write_text(
        "[Material]\n"
        "name=Transport\n"
        "[Data]\n"
        "wavelength_nm n k\n"
        "400 1.8 0.01\n"
        "600 1.9 0.02\n",
        encoding="utf-8",
    )

    data = load_nk_file(path)

    assert data.name == "transport"
    assert np.allclose(data.complex_index([500.0]), [1.85 + 0.015j])


def test_lumerical_binary_mdf_file_is_supported():
    data = load_nk_file("examples/materials/1060-hl.mdf")

    assert data.name == "1060"
    assert len(data.wavelength_nm) == 951
    assert 699.0 < data.min_wavelength_nm < 701.0
    assert 1649.0 < data.max_wavelength_nm < 1651.0
    values = data.complex_index([700.0, 1060.0, 1650.0])
    assert np.all(values.real > 0.0)
    assert np.all(values.imag >= 0.0)


def test_tabulated_data_can_extrapolate_when_requested(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text(
        "wavelength,n,k\n"
        "500,2.0,0.10\n"
        "600,2.2,0.20\n",
        encoding="utf-8",
    )

    data = load_nk_file(path)

    assert np.allclose(data.complex_index([700.0], bounds_policy="extrapolate"), [2.4 + 0.30j])
