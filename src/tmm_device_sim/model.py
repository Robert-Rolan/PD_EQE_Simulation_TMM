from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .materials import OpticalData


@dataclass(frozen=True)
class Layer:
    name: str
    thickness_nm: float
    optical_data: OpticalData
    color: str = "#9fb6d8"
    active: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("layer name cannot be empty")
        if self.thickness_nm <= 0:
            raise ValueError("layer thickness must be positive")

    def with_thickness(self, thickness_nm: float) -> "Layer":
        return replace(self, thickness_nm=float(thickness_nm))


@dataclass(frozen=True)
class Medium:
    name: str
    optical_data: OpticalData

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("medium name cannot be empty")


def wavelength_grid(start_nm: float = 300.0, stop_nm: float = 1200.0, step_nm: float = 5.0) -> np.ndarray:
    if step_nm <= 0:
        raise ValueError("wavelength step must be positive")
    if stop_nm < start_nm:
        raise ValueError("wavelength stop must be greater than or equal to start")
    count = int(np.floor((stop_nm - start_nm) / step_nm + 0.5)) + 1
    values = start_nm + np.arange(count, dtype=float) * step_nm
    if values[-1] < stop_nm - 1e-9:
        values = np.append(values, stop_nm)
    return values


def total_thickness_nm(layers: Iterable[Layer]) -> float:
    return float(sum(layer.thickness_nm for layer in layers))
