from __future__ import annotations

from pathlib import Path
import csv

from .batch import BatchResult
from .simulation import SimulationResult


def export_spectrum_csv(result: SimulationResult, path: str | Path) -> None:
    columns = ["wavelength_nm", "reflectance", "transmittance", "total_absorption", "ideal_eqe"]
    layer_names = list(result.layer_absorption)
    columns.extend(f"absorption_{name}" for name in layer_names)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, wavelength in enumerate(result.wavelength_nm):
            writer.writerow(
                [
                    wavelength,
                    result.reflectance[index],
                    result.transmittance[index],
                    result.total_absorption[index],
                    result.eqe[index],
                    *[result.layer_absorption[name][index] for name in layer_names],
                ]
            )


def export_field_csv(result: SimulationResult, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["depth_nm", *[f"{wavelength:g}_nm" for wavelength in result.wavelength_nm]])
        for depth_index, depth in enumerate(result.field_depth_nm):
            writer.writerow([depth, *result.field_intensity[depth_index, :]])


def export_batch_csv(result: BatchResult, path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["thickness_nm", *[f"{wavelength:g}_nm" for wavelength in result.wavelength_nm]])
        for thickness_index, thickness in enumerate(result.thicknesses_nm):
            writer.writerow([thickness, *result.eqe_map[thickness_index, :]])
