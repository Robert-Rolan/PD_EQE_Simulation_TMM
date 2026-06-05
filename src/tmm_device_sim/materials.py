from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct
from typing import Iterable, Literal, Protocol

import numpy as np


class OpticalDataError(ValueError):
    """Raised when optical constant data cannot be used safely."""


BoundsPolicy = Literal["raise", "clip", "extrapolate"]
WavelengthUnit = Literal["auto", "nm", "um"]
LUMERICAL_MDF_MAGIC = b"Lumerical material data file version"
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


class OpticalData(Protocol):
    def complex_index(
        self,
        wavelengths_nm: Iterable[float] | np.ndarray,
        bounds_policy: BoundsPolicy = "raise",
    ) -> np.ndarray:
        """Return complex refractive index n + i k at wavelengths in nm."""


@dataclass(frozen=True)
class ConstantOpticalData:
    n: float
    k: float = 0.0

    def complex_index(
        self,
        wavelengths_nm: Iterable[float] | np.ndarray,
        bounds_policy: BoundsPolicy = "raise",
    ) -> np.ndarray:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        return np.full(wavelengths.shape, complex(self.n, self.k), dtype=complex)


@dataclass(frozen=True)
class TabulatedOpticalData:
    wavelength_nm: np.ndarray
    n: np.ndarray
    k: np.ndarray
    name: str = ""
    source_wavelength: np.ndarray | None = None
    wavelength_unit: str = "nm"

    def __post_init__(self) -> None:
        wavelength = np.asarray(self.wavelength_nm, dtype=float)
        n_values = np.asarray(self.n, dtype=float)
        k_values = np.asarray(self.k, dtype=float)
        unit = _normalize_wavelength_unit(self.wavelength_unit)
        source_wavelength = (
            np.asarray(self.source_wavelength, dtype=float)
            if self.source_wavelength is not None
            else wavelength / _unit_multiplier(unit)
        )
        if wavelength.ndim != 1 or n_values.ndim != 1 or k_values.ndim != 1:
            raise OpticalDataError("wavelength, n, and k columns must be one-dimensional")
        if source_wavelength.ndim != 1:
            raise OpticalDataError("source wavelength column must be one-dimensional")
        if not (len(wavelength) == len(n_values) == len(k_values) == len(source_wavelength)):
            raise OpticalDataError("wavelength, n, and k columns must have the same length")
        if len(wavelength) < 2:
            raise OpticalDataError("at least two wavelength rows are required")
        if (
            np.any(~np.isfinite(wavelength))
            or np.any(~np.isfinite(n_values))
            or np.any(~np.isfinite(k_values))
            or np.any(~np.isfinite(source_wavelength))
        ):
            raise OpticalDataError("wavelength, n, and k data must be finite numbers")
        if np.any(wavelength <= 0):
            raise OpticalDataError("wavelength values must be positive")

        order = np.argsort(wavelength)
        wavelength = wavelength[order]
        n_values = n_values[order]
        k_values = k_values[order]
        source_wavelength = source_wavelength[order]
        if np.any(np.diff(wavelength) <= 0):
            raise OpticalDataError("wavelength values must be unique")

        object.__setattr__(self, "wavelength_nm", wavelength)
        object.__setattr__(self, "n", n_values)
        object.__setattr__(self, "k", k_values)
        object.__setattr__(self, "source_wavelength", source_wavelength)
        object.__setattr__(self, "wavelength_unit", unit)

    @property
    def min_wavelength_nm(self) -> float:
        return float(self.wavelength_nm[0])

    @property
    def max_wavelength_nm(self) -> float:
        return float(self.wavelength_nm[-1])

    def with_wavelength_unit(self, wavelength_unit: str) -> "TabulatedOpticalData":
        unit = _normalize_wavelength_unit(wavelength_unit)
        if self.source_wavelength is None:
            source_wavelength = self.wavelength_nm / _unit_multiplier(self.wavelength_unit)
        else:
            source_wavelength = self.source_wavelength
        return TabulatedOpticalData(
            wavelength_nm=source_wavelength * _unit_multiplier(unit),
            n=self.n,
            k=self.k,
            name=self.name,
            source_wavelength=source_wavelength,
            wavelength_unit=unit,
        )

    def complex_index(
        self,
        wavelengths_nm: Iterable[float] | np.ndarray,
        bounds_policy: BoundsPolicy = "raise",
    ) -> np.ndarray:
        policy = _normalize_bounds_policy(bounds_policy)
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if np.any(~np.isfinite(wavelengths)):
            raise OpticalDataError("requested wavelengths must be finite")
        min_requested = float(np.min(wavelengths))
        max_requested = float(np.max(wavelengths))
        tolerance = 1e-9
        if min_requested < self.min_wavelength_nm - tolerance or max_requested > self.max_wavelength_nm + tolerance:
            if policy == "raise":
                raise OpticalDataError(
                    f"requested wavelength range {min_requested:g}-{max_requested:g} nm is outside "
                    f"available range {self.min_wavelength_nm:g}-{self.max_wavelength_nm:g} nm"
                )
            if policy == "clip":
                wavelengths = np.clip(wavelengths, self.min_wavelength_nm, self.max_wavelength_nm)
        if policy == "extrapolate":
            n_values = _interp_with_linear_extrapolation(wavelengths, self.wavelength_nm, self.n)
            k_values = _interp_with_linear_extrapolation(wavelengths, self.wavelength_nm, self.k)
        else:
            n_values = np.interp(wavelengths, self.wavelength_nm, self.n)
            k_values = np.interp(wavelengths, self.wavelength_nm, self.k)
        return n_values + 1j * k_values


def load_nk_file(path: str | Path, wavelength_unit: WavelengthUnit = "auto") -> TabulatedOpticalData:
    file_path = Path(path)
    raw = file_path.read_bytes()
    if file_path.suffix.lower() == ".mdf" and raw.startswith(LUMERICAL_MDF_MAGIC):
        return _load_lumerical_binary_mdf(raw, file_path)

    text = _decode_text_file(raw, file_path)
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if len(lines) < 2:
        raise OpticalDataError(f"{file_path} must contain one header row and at least one data row")

    header_index, headers, column_map = _find_header(lines)
    unit = (
        _normalize_wavelength_unit(wavelength_unit)
        if wavelength_unit != "auto"
        else _unit_from_header(headers[column_map["wavelength"]])
    )
    wavelength_multiplier = _unit_multiplier(unit)

    rows: list[list[float]] = []
    for line in lines[header_index + 1 :]:
        parts = _split_row(line)
        if len(parts) <= max(column_map.values()):
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError as exc:
            if file_path.suffix.lower() == ".mdf":
                continue
            raise OpticalDataError(f"data row contains a non-numeric value: {line}") from exc
    if not rows:
        raise OpticalDataError(f"{file_path} contains no numeric wavelength, n, k rows")

    data = np.asarray(rows, dtype=float)
    source_wavelength = data[:, column_map["wavelength"]]
    return TabulatedOpticalData(
        wavelength_nm=source_wavelength * wavelength_multiplier,
        n=data[:, column_map["n"]],
        k=data[:, column_map["k"]],
        name=file_path.stem,
        source_wavelength=source_wavelength,
        wavelength_unit=unit,
    )


def builtin_optical_data(name: str) -> ConstantOpticalData:
    key = _normalize_medium_name(name)
    if key in {"air", "vacuum"}:
        return ConstantOpticalData(1.0, 0.0)
    if key == "glass":
        return ConstantOpticalData(1.5, 0.0)
    raise OpticalDataError(f"{name} needs imported n,k data before it can be used as a medium")


def _split_row(line: str) -> list[str]:
    return [part for part in re.split(r"[,\s;]+", line.strip()) if part]


def _decode_text_file(raw: bytes, file_path: Path) -> str:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise OpticalDataError(f"could not decode {file_path} as text; " + "; ".join(errors))


def _normalize_header(header: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", header.lower())


def _identify_columns(headers: list[str]) -> dict[str, int]:
    normalized = [_normalize_header(header) for header in headers]

    wavelength_index = _find_column(
        normalized,
        lambda value: "wavelength" in value or value in {"lambda", "lambda0", "wl", "lam"},
        "wavelength",
    )
    n_index = _find_column(
        normalized,
        lambda value: value in {"n", "realn", "index"} or "refractive" in value,
        "n",
    )
    k_index = _find_column(
        normalized,
        lambda value: value in {"k", "imag", "imaginary"} or "extinction" in value,
        "k",
    )
    return {"wavelength": wavelength_index, "n": n_index, "k": k_index}


def _find_header(lines: list[str]) -> tuple[int, list[str], dict[str, int]]:
    first_error: OpticalDataError | None = None
    for index, line in enumerate(lines):
        headers = _split_row(line)
        try:
            return index, headers, _identify_columns(headers)
        except OpticalDataError as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
    raise OpticalDataError("could not find wavelength, n, k header")


def _find_column(headers: list[str], predicate, label: str) -> int:
    for index, header in enumerate(headers):
        if predicate(header):
            return index
    raise OpticalDataError(f"could not find required {label} column in n,k file header")


def _unit_from_header(header: str) -> str:
    normalized = _normalize_header(header)
    if "um" in normalized or "micron" in normalized:
        return "um"
    if "ev" in normalized:
        raise OpticalDataError("energy-domain optical constants are not supported; provide wavelength in nm")
    return "nm"


def _normalize_wavelength_unit(wavelength_unit: str) -> str:
    normalized = wavelength_unit.strip().lower().replace("\u03bc", "u").replace("\u00b5", "u")
    if normalized in {"nm", "nanometer", "nanometers"}:
        return "nm"
    if normalized in {"um", "micron", "microns", "micrometer", "micrometers"}:
        return "um"
    raise OpticalDataError("wavelength unit must be 'nm' or 'um'")


def _unit_multiplier(wavelength_unit: str) -> float:
    unit = _normalize_wavelength_unit(wavelength_unit)
    return 1000.0 if unit == "um" else 1.0


def _normalize_bounds_policy(bounds_policy: BoundsPolicy) -> BoundsPolicy:
    if bounds_policy not in {"raise", "clip", "extrapolate"}:
        raise OpticalDataError("bounds policy must be 'raise', 'clip', or 'extrapolate'")
    return bounds_policy


def _interp_with_linear_extrapolation(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    values = np.interp(x, xp, fp)
    low = x < xp[0]
    high = x > xp[-1]
    if np.any(low):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        values[low] = fp[0] + slope * (x[low] - xp[0])
    if np.any(high):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        values[high] = fp[-1] + slope * (x[high] - xp[-1])
    return values


def _normalize_medium_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _load_lumerical_binary_mdf(raw: bytes, file_path: Path) -> TabulatedOpticalData:
    material = _select_lumerical_material(raw, file_path.stem)
    frequency = _parse_lumerical_array(raw, material["frequency_pos"], b"frequency")
    permittivity = _parse_lumerical_array(raw, material["permittivity_pos"], b"permittivity")

    if np.iscomplexobj(frequency):
        frequency = np.real(frequency)
    frequency = np.asarray(frequency, dtype=float)
    permittivity = np.asarray(permittivity, dtype=complex)
    if len(frequency) != len(permittivity):
        raise OpticalDataError(
            f"Lumerical MDF material '{material['name']}' has mismatched frequency and permittivity lengths"
        )
    if np.any(frequency <= 0) or np.any(~np.isfinite(frequency)):
        raise OpticalDataError(f"Lumerical MDF material '{material['name']}' contains invalid frequency values")
    if np.any(~np.isfinite(permittivity.real)) or np.any(~np.isfinite(permittivity.imag)):
        raise OpticalDataError(f"Lumerical MDF material '{material['name']}' contains invalid permittivity values")

    wavelength_nm = SPEED_OF_LIGHT_M_PER_S / frequency * 1e9
    refractive_index = np.sqrt(permittivity)
    n_values = np.real(refractive_index)
    k_values = np.abs(np.imag(refractive_index))
    if np.any(n_values < 0):
        refractive_index = -refractive_index
        n_values = np.real(refractive_index)
        k_values = np.abs(np.imag(refractive_index))

    return TabulatedOpticalData(
        wavelength_nm=wavelength_nm,
        n=n_values,
        k=k_values,
        name=str(material["name"]),
        source_wavelength=wavelength_nm,
        wavelength_unit="nm",
    )


def _select_lumerical_material(raw: bytes, file_stem: str) -> dict[str, int | str]:
    names = _lumerical_material_names(raw)
    candidates = _material_name_candidates(file_stem)
    matching = [item for item in names if _normalize_material_key(str(item["name"])) in candidates]
    if not matching:
        matching = [
            item
            for item in names
            if any(candidate and candidate in _normalize_material_key(str(item["name"])) for candidate in candidates)
        ]
    if not matching:
        usable = [item for item in names if _has_lumerical_optical_arrays(raw, int(item["name_pos"]))]
        if len(usable) == 1:
            matching = usable
        else:
            shown = ", ".join(str(item["name"]) for item in usable[:10])
            raise OpticalDataError(
                f"could not choose a material from binary MDF '{file_stem}'. "
                f"Rename the file to match one material name. Available examples: {shown}"
            )

    for item in matching:
        frequency_pos = raw.rfind(b"frequency\x00", 0, int(item["name_pos"]))
        permittivity_pos = raw.find(b"permittivity\x00", int(item["name_pos"]), int(item["name_pos"]) + 50000)
        if frequency_pos >= 0 and permittivity_pos >= 0:
            return {
                "name": str(item["name"]),
                "name_pos": int(item["name_pos"]),
                "frequency_pos": frequency_pos,
                "permittivity_pos": permittivity_pos,
            }
    raise OpticalDataError(f"binary MDF material '{matching[0]['name']}' does not contain frequency/permittivity data")


def _lumerical_material_names(raw: bytes) -> list[dict[str, int | str]]:
    items: list[dict[str, int | str]] = []
    for match in re.finditer(b"name\x00", raw):
        pos = match.start()
        if pos < 4 or _read_u32(raw, pos - 4) != len(b"name\x00"):
            continue
        value = _parse_lumerical_string_value(raw, pos + len(b"name\x00"))
        if value:
            items.append({"name": value, "name_pos": pos})
    return items


def _parse_lumerical_string_value(raw: bytes, pos: int) -> str | None:
    if pos + 8 > len(raw) or _read_u32(raw, pos) != 0:
        return None
    length = _read_u32(raw, pos + 4)
    if length <= 1 or pos + 8 + length > len(raw):
        return None
    value = raw[pos + 8 : pos + 8 + length].rstrip(b"\x00")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("latin-1", errors="replace")


def _parse_lumerical_array(raw: bytes, prop_pos: int, prop_name: bytes) -> np.ndarray:
    pos = prop_pos + len(prop_name) + 1
    if pos + 32 > len(raw) or _read_u32(raw, pos) != 3:
        raise OpticalDataError(f"binary MDF property '{prop_name.decode('ascii')}' is not a numeric array")

    count = _read_u32(raw, pos + 8)
    scalar_type = _read_u32(raw, pos + 12)
    is_real = _read_u32(raw, pos + 24)
    if count <= 0 or scalar_type != 2:
        raise OpticalDataError(f"binary MDF property '{prop_name.decode('ascii')}' has an unsupported array header")

    components = 1 if is_real else 2
    data_start = pos + 32
    data_count = count * components
    data_end = data_start + data_count * 8
    if data_end > len(raw):
        raise OpticalDataError(f"binary MDF property '{prop_name.decode('ascii')}' data is truncated")

    values = np.frombuffer(raw, dtype="<f8", count=data_count, offset=data_start).astype(float, copy=True)
    if components == 1:
        return values
    return values[:count] + 1j * values[count:]


def _has_lumerical_optical_arrays(raw: bytes, name_pos: int) -> bool:
    frequency_pos = raw.rfind(b"frequency\x00", 0, name_pos)
    permittivity_pos = raw.find(b"permittivity\x00", name_pos, name_pos + 50000)
    return frequency_pos >= 0 and permittivity_pos >= 0


def _material_name_candidates(file_stem: str) -> set[str]:
    normalized = _normalize_material_key(file_stem)
    parts = {_normalize_material_key(part) for part in re.split(r"[-_\s]+", file_stem) if part}
    return {normalized, *parts}


def _normalize_material_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _read_u32(raw: bytes, pos: int) -> int:
    if pos < 0 or pos + 4 > len(raw):
        return 0
    return struct.unpack_from("<I", raw, pos)[0]
