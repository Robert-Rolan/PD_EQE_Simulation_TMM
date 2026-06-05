from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .materials import BoundsPolicy, OpticalData, OpticalDataError
from .model import Layer, Medium, total_thickness_nm

Incidence = Literal["top", "bottom"]
FieldScope = Literal["active", "stack", "custom"]


@dataclass(frozen=True)
class SimulationResult:
    wavelength_nm: np.ndarray
    reflectance: np.ndarray
    transmittance: np.ndarray
    total_absorption: np.ndarray
    eqe: np.ndarray
    layer_absorption: dict[str, np.ndarray]
    field_depth_nm: np.ndarray
    field_intensity: np.ndarray
    incidence: Incidence
    layers: tuple[Layer, ...]


def simulate_stack(
    layers: list[Layer] | tuple[Layer, ...],
    *,
    bottom_medium: Medium,
    top_medium: Medium,
    wavelengths_nm,
    incidence: Incidence = "top",
    field_scope: FieldScope = "active",
    custom_depth_range_nm: tuple[float, float] | None = None,
    depth_step_nm: float = 2.0,
    wavelength_bounds_policy: BoundsPolicy = "raise",
) -> SimulationResult:
    physical_layers = tuple(layers)
    if not physical_layers:
        raise ValueError("at least one layer is required")
    if incidence not in {"top", "bottom"}:
        raise ValueError("incidence must be 'top' or 'bottom'")
    if depth_step_nm <= 0:
        raise ValueError("depth step must be positive")

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1 or len(wavelengths) == 0:
        raise ValueError("wavelengths must be a non-empty one-dimensional array")
    wavelengths = _apply_wavelength_bounds_policy(
        physical_layers,
        (bottom_medium, top_medium),
        wavelengths,
        wavelength_bounds_policy,
    )

    ordered_layers, physical_to_ordered, ordered_to_physical = _ordered_layers(physical_layers, incidence)
    incident_medium = top_medium if incidence == "top" else bottom_medium
    exit_medium = bottom_medium if incidence == "top" else top_medium
    depths = _depth_samples(physical_layers, field_scope, custom_depth_range_nm, depth_step_nm)

    reflectance = np.zeros_like(wavelengths, dtype=float)
    transmittance = np.zeros_like(wavelengths, dtype=float)
    total_absorption = np.zeros_like(wavelengths, dtype=float)
    eqe = np.zeros_like(wavelengths, dtype=float)
    layer_absorption = {layer.name: np.zeros_like(wavelengths, dtype=float) for layer in physical_layers}
    field_intensity = np.zeros((len(depths), len(wavelengths)), dtype=float)
    boundaries = _physical_boundaries(physical_layers)

    for wavelength_index, wavelength in enumerate(wavelengths):
        n_in = _complex_index(incident_medium.optical_data, [wavelength], wavelength_bounds_policy)[0]
        n_out = _complex_index(exit_medium.optical_data, [wavelength], wavelength_bounds_policy)[0]
        n_layers = np.array(
            [_complex_index(layer.optical_data, [wavelength], wavelength_bounds_policy)[0] for layer in ordered_layers],
            dtype=complex,
        )
        r, t, forward, backward = _solve_amplitudes(ordered_layers, n_layers, n_in, n_out, wavelength)

        r_value = abs(r) ** 2
        t_value = max(0.0, float(np.real(n_out) / np.real(n_in) * abs(t) ** 2))
        a_value = float(np.real(1.0 - r_value - t_value))
        if abs(a_value) < 1e-10:
            a_value = 0.0

        reflectance[wavelength_index] = r_value
        transmittance[wavelength_index] = t_value
        total_absorption[wavelength_index] = a_value

        for ordered_index, layer in enumerate(ordered_layers):
            physical_index = ordered_to_physical[ordered_index]
            flux_start = _poynting_flux(
                n_layers[ordered_index],
                wavelength,
                0.0,
                forward[ordered_index],
                backward[ordered_index],
                np.real(n_in),
            )
            flux_end = _poynting_flux(
                n_layers[ordered_index],
                wavelength,
                layer.thickness_nm,
                forward[ordered_index],
                backward[ordered_index],
                np.real(n_in),
            )
            absorption = float(flux_start - flux_end)
            if abs(absorption) < 1e-10:
                absorption = 0.0
            layer_absorption[physical_layers[physical_index].name][wavelength_index] = max(0.0, absorption)

        active_absorption = [
            layer_absorption[layer.name][wavelength_index]
            for layer in physical_layers
            if layer.active
        ]
        eqe[wavelength_index] = float(np.sum(active_absorption)) if active_absorption else 0.0

        for depth_index, depth in enumerate(depths):
            physical_index, local_from_bottom = _locate_depth(boundaries, depth)
            ordered_index = physical_to_ordered[physical_index]
            physical_layer = physical_layers[physical_index]
            if incidence == "bottom":
                local_from_incident_side = local_from_bottom
            else:
                local_from_incident_side = physical_layer.thickness_nm - local_from_bottom
            electric_field = _field_value(
                n_layers[ordered_index],
                wavelength,
                local_from_incident_side,
                forward[ordered_index],
                backward[ordered_index],
            )
            field_intensity[depth_index, wavelength_index] = abs(electric_field) ** 2

    return SimulationResult(
        wavelength_nm=wavelengths,
        reflectance=reflectance,
        transmittance=transmittance,
        total_absorption=total_absorption,
        eqe=eqe,
        layer_absorption=layer_absorption,
        field_depth_nm=depths,
        field_intensity=field_intensity,
        incidence=incidence,
        layers=physical_layers,
    )


def _ordered_layers(layers: tuple[Layer, ...], incidence: Incidence):
    if incidence == "bottom":
        ordered = layers
        physical_to_ordered = {index: index for index in range(len(layers))}
        ordered_to_physical = {index: index for index in range(len(layers))}
    else:
        ordered = tuple(reversed(layers))
        physical_to_ordered = {physical_index: len(layers) - 1 - physical_index for physical_index in range(len(layers))}
        ordered_to_physical = {ordered_index: len(layers) - 1 - ordered_index for ordered_index in range(len(layers))}
    return ordered, physical_to_ordered, ordered_to_physical


def _apply_wavelength_bounds_policy(
    layers: tuple[Layer, ...],
    media: tuple[Medium, Medium],
    wavelengths: np.ndarray,
    bounds_policy: BoundsPolicy,
) -> np.ndarray:
    if bounds_policy not in {"raise", "clip", "extrapolate"}:
        raise OpticalDataError("bounds policy must be 'raise', 'clip', or 'extrapolate'")
    if bounds_policy != "clip":
        return wavelengths

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for optical_data in [layer.optical_data for layer in layers] + [medium.optical_data for medium in media]:
        coverage = _coverage(optical_data)
        if coverage is not None:
            lower_bounds.append(coverage[0])
            upper_bounds.append(coverage[1])
    if not lower_bounds:
        return wavelengths

    lower = max(lower_bounds)
    upper = min(upper_bounds)
    if lower > upper:
        raise OpticalDataError("no overlapping wavelength range exists across imported n,k data")

    mask = (wavelengths >= lower - 1e-9) & (wavelengths <= upper + 1e-9)
    clipped = wavelengths[mask]
    if len(clipped) == 0:
        raise OpticalDataError(
            f"no requested wavelengths fall inside available n,k range {lower:g}-{upper:g} nm"
        )
    return clipped


def _coverage(optical_data: OpticalData) -> tuple[float, float] | None:
    minimum = getattr(optical_data, "min_wavelength_nm", None)
    maximum = getattr(optical_data, "max_wavelength_nm", None)
    if minimum is None or maximum is None:
        return None
    return float(minimum), float(maximum)


def _complex_index(
    optical_data: OpticalData,
    wavelengths_nm,
    bounds_policy: BoundsPolicy,
) -> np.ndarray:
    return optical_data.complex_index(wavelengths_nm, bounds_policy=bounds_policy)


def _solve_amplitudes(
    layers: tuple[Layer, ...],
    n_layers: np.ndarray,
    n_in: complex,
    n_out: complex,
    wavelength_nm: float,
) -> tuple[complex, complex, np.ndarray, np.ndarray]:
    layer_count = len(layers)
    if layer_count == 0:
        r = (n_in - n_out) / (n_in + n_out)
        t = 2 * n_in / (n_in + n_out)
        return r, t, np.array([], dtype=complex), np.array([], dtype=complex)

    variable_count = 2 * layer_count + 2
    matrix = np.zeros((variable_count, variable_count), dtype=complex)
    rhs = np.zeros(variable_count, dtype=complex)

    r_index = 0
    t_index = variable_count - 1

    def a_index(layer_index: int) -> int:
        return 1 + 2 * layer_index

    def b_index(layer_index: int) -> int:
        return 2 + 2 * layer_index

    row = 0
    matrix[row, r_index] = 1.0
    matrix[row, a_index(0)] = -1.0
    matrix[row, b_index(0)] = -1.0
    rhs[row] = -1.0
    row += 1

    matrix[row, r_index] = -n_in
    matrix[row, a_index(0)] = -n_layers[0]
    matrix[row, b_index(0)] = n_layers[0]
    rhs[row] = -n_in
    row += 1

    for layer_index in range(layer_count - 1):
        phase = _phase(n_layers[layer_index], layers[layer_index].thickness_nm, wavelength_nm)
        inv_phase = 1.0 / phase
        next_index = layer_index + 1

        matrix[row, a_index(layer_index)] = phase
        matrix[row, b_index(layer_index)] = inv_phase
        matrix[row, a_index(next_index)] = -1.0
        matrix[row, b_index(next_index)] = -1.0
        row += 1

        matrix[row, a_index(layer_index)] = n_layers[layer_index] * phase
        matrix[row, b_index(layer_index)] = -n_layers[layer_index] * inv_phase
        matrix[row, a_index(next_index)] = -n_layers[next_index]
        matrix[row, b_index(next_index)] = n_layers[next_index]
        row += 1

    last_index = layer_count - 1
    phase = _phase(n_layers[last_index], layers[last_index].thickness_nm, wavelength_nm)
    inv_phase = 1.0 / phase
    matrix[row, a_index(last_index)] = phase
    matrix[row, b_index(last_index)] = inv_phase
    matrix[row, t_index] = -1.0
    row += 1

    matrix[row, a_index(last_index)] = n_layers[last_index] * phase
    matrix[row, b_index(last_index)] = -n_layers[last_index] * inv_phase
    matrix[row, t_index] = -n_out

    solution = np.linalg.solve(matrix, rhs)
    forward = np.array([solution[a_index(index)] for index in range(layer_count)], dtype=complex)
    backward = np.array([solution[b_index(index)] for index in range(layer_count)], dtype=complex)
    return solution[r_index], solution[t_index], forward, backward


def _phase(n_value: complex, thickness_nm: float, wavelength_nm: float) -> complex:
    return np.exp(1j * 2.0 * np.pi * n_value * thickness_nm / wavelength_nm)


def _field_value(n_value: complex, wavelength_nm: float, z_nm: float, forward: complex, backward: complex) -> complex:
    phase = _phase(n_value, z_nm, wavelength_nm)
    return forward * phase + backward / phase


def _poynting_flux(
    n_value: complex,
    wavelength_nm: float,
    z_nm: float,
    forward: complex,
    backward: complex,
    incident_n_real: float,
) -> float:
    phase = _phase(n_value, z_nm, wavelength_nm)
    electric = forward * phase + backward / phase
    magnetic = n_value * (forward * phase - backward / phase)
    return float(np.real(electric * np.conjugate(magnetic)) / incident_n_real)


def _depth_samples(
    layers: tuple[Layer, ...],
    field_scope: FieldScope,
    custom_range: tuple[float, float] | None,
    depth_step_nm: float,
) -> np.ndarray:
    total = total_thickness_nm(layers)
    if field_scope == "stack":
        start, stop = 0.0, total
    elif field_scope == "custom":
        if custom_range is None:
            raise ValueError("custom field range needs start and stop depths")
        start, stop = custom_range
        if start < 0 or stop > total or stop < start:
            raise ValueError("custom field range must lie within the layer stack")
    else:
        active_ranges = _active_ranges(layers)
        if active_ranges:
            start = min(item[0] for item in active_ranges)
            stop = max(item[1] for item in active_ranges)
        else:
            start, stop = 0.0, total

    if np.isclose(start, stop):
        return np.array([start], dtype=float)
    count = int(np.ceil((stop - start) / depth_step_nm))
    depths = start + np.arange(count + 1, dtype=float) * depth_step_nm
    depths[-1] = stop
    return depths


def _active_ranges(layers: tuple[Layer, ...]) -> list[tuple[float, float]]:
    ranges = []
    cursor = 0.0
    for layer in layers:
        next_cursor = cursor + layer.thickness_nm
        if layer.active:
            ranges.append((cursor, next_cursor))
        cursor = next_cursor
    return ranges


def _physical_boundaries(layers: tuple[Layer, ...]) -> np.ndarray:
    boundaries = [0.0]
    for layer in layers:
        boundaries.append(boundaries[-1] + layer.thickness_nm)
    return np.asarray(boundaries, dtype=float)


def _locate_depth(boundaries: np.ndarray, depth_nm: float) -> tuple[int, float]:
    if np.isclose(depth_nm, boundaries[-1]):
        index = len(boundaries) - 2
    else:
        index = int(np.searchsorted(boundaries, depth_nm, side="right") - 1)
        index = max(0, min(index, len(boundaries) - 2))
    local = float(depth_nm - boundaries[index])
    return index, local
