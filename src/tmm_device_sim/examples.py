from __future__ import annotations

from .materials import ConstantOpticalData
from .model import Layer, Medium


def default_media() -> tuple[Medium, Medium]:
    return Medium("glass", ConstantOpticalData(1.5, 0.0)), Medium("air", ConstantOpticalData(1.0, 0.0))


def default_layers() -> list[Layer]:
    return [
        Layer("ITO", 130.0, ConstantOpticalData(1.8, 0.02), "#a9dff0"),
        Layer("ZnO", 40.0, ConstantOpticalData(2.0, 0.01), "#c7d8f2"),
        Layer("BHJ", 280.0, ConstantOpticalData(2.1, 0.12), "#8999b8", active=True),
        Layer("MoOx", 7.0, ConstantOpticalData(2.0, 0.05), "#d8d8e8"),
        Layer("Ag", 100.0, ConstantOpticalData(0.15, 3.5), "#858585"),
    ]
