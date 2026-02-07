from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VisualScene:
    title: str
    description: str


def build_math_concept_scene() -> VisualScene:
    return VisualScene(
        title="Fraction Bars",
        description="Interactive bars show parts of a whole to ground abstract fractions.",
    )
