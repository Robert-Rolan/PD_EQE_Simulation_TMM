from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import BoundsPolicy
from .model import Layer, Medium
from .simulation import Incidence, SimulationResult, simulate_stack


@dataclass(frozen=True)
class ThicknessSweep:
    layer_index: int
    start_nm: float
    stop_nm: float
    step_nm: float

    def values(self) -> np.ndarray:
        if self.layer_index < 0:
            raise ValueError("layer index must be non-negative")
        if self.step_nm <= 0:
            raise ValueError("thickness sweep step must be positive")
        if self.stop_nm < self.start_nm:
            raise ValueError("thickness sweep stop must be greater than or equal to start")
        count = int(np.floor((self.stop_nm - self.start_nm) / self.step_nm + 0.5)) + 1
        values = self.start_nm + np.arange(count, dtype=float) * self.step_nm
        if values[-1] < self.stop_nm - 1e-9:
            values = np.append(values, self.stop_nm)
        return values


@dataclass(frozen=True)
class BatchResult:
    thicknesses_nm: np.ndarray
    wavelength_nm: np.ndarray
    eqe_map: np.ndarray
    results: tuple[SimulationResult, ...]


def eqe_at_wavelength_vs_thickness(result: BatchResult, wavelength_nm: float) -> np.ndarray:
    if len(result.wavelength_nm) == 0:
        raise ValueError("batch result has no wavelength data")
    if wavelength_nm < result.wavelength_nm[0] or wavelength_nm > result.wavelength_nm[-1]:
        raise ValueError(
            f"trace wavelength {wavelength_nm:g} nm is outside batch range "
            f"{result.wavelength_nm[0]:g}-{result.wavelength_nm[-1]:g} nm"
        )
    return np.array(
        [
            np.interp(float(wavelength_nm), result.wavelength_nm, result.eqe_map[thickness_index, :])
            for thickness_index in range(len(result.thicknesses_nm))
        ],
        dtype=float,
    )


def run_thickness_sweep(
    layers: list[Layer] | tuple[Layer, ...],
    *,
    bottom_medium: Medium,
    top_medium: Medium,
    wavelengths_nm,
    sweep: ThicknessSweep,
    incidence: Incidence = "top",
    wavelength_bounds_policy: BoundsPolicy = "raise",
) -> BatchResult:
    base_layers = tuple(layers)
    if sweep.layer_index >= len(base_layers):
        raise IndexError("thickness sweep layer index is outside the stack")

    thicknesses = sweep.values()
    results: list[SimulationResult] = []
    for thickness in thicknesses:
        swept_layers = list(base_layers)
        swept_layers[sweep.layer_index] = swept_layers[sweep.layer_index].with_thickness(float(thickness))
        results.append(
            simulate_stack(
                swept_layers,
                bottom_medium=bottom_medium,
                top_medium=top_medium,
                wavelengths_nm=wavelengths_nm,
                incidence=incidence,
                wavelength_bounds_policy=wavelength_bounds_policy,
            )
        )

    eqe_map = np.vstack([result.eqe for result in results])
    return BatchResult(
        thicknesses_nm=thicknesses,
        wavelength_nm=results[0].wavelength_nm.copy(),
        eqe_map=eqe_map,
        results=tuple(results),
    )
