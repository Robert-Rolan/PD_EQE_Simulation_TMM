# PD_EQE_Simulation_TMM

Desktop Python tool for transfer-matrix simulation of thin-film photodiode EQE spectra from measured n and k optical parameters, with field-intensity visualization and thickness-dependent EQE analysis.

## Features

- Build multilayer photodiode stacks with active-layer selection.
- Simulate ideal EQE, reflectance, transmittance, absorption, and field intensity.
- Compare top and bottom incidence.
- Run active-layer or transport-layer thickness sweeps.
- Import optical constants from CSV, TXT, or Lumerical MDF files.
- Export spectra, field maps, batch EQE tables, and figures.

## Install

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run

From the project root:

```powershell
python -m tmm_device_sim
```

After editable installation, this command is also available:

```powershell
tmm-device-sim
```

On Windows, you can also run `start_tmm_device_sim.bat`.

## Test

```powershell
python -m pytest
```

## Optical Data

The GUI can import n,k files with wavelength, n, and k columns. Wavelengths may be in nm or um, and supported file types include CSV, TXT, and selected Lumerical MDF files.

Example material files are included in `examples/materials/` for testing and demonstration. Personal measurement data and local reference PDFs are not included in the repository.
