"""Thin-film photodiode TMM simulator."""

from .batch import ThicknessSweep, eqe_at_wavelength_vs_thickness, run_thickness_sweep
from .materials import ConstantOpticalData, OpticalDataError, TabulatedOpticalData, load_nk_file
from .model import Layer, Medium, wavelength_grid
from .simulation import SimulationResult, simulate_stack

__all__ = [
    "ConstantOpticalData",
    "Layer",
    "Medium",
    "OpticalDataError",
    "SimulationResult",
    "TabulatedOpticalData",
    "ThicknessSweep",
    "load_nk_file",
    "eqe_at_wavelength_vs_thickness",
    "run_thickness_sweep",
    "simulate_stack",
    "wavelength_grid",
]
