from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class VisualScene:
    title: str
    description: str


@dataclass
class FractionBar:
    numerator: int
    denominator: int
    color: str
    
    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator != 0 else 0.0
    
    @property
    def filled_parts(self) -> int:
        return self.numerator
    
    @property
    def total_parts(self) -> int:
        return self.denominator
    
    def to_text(self) -> str:
        return f"{self.numerator}/{self.denominator}"
    
    def to_spoken_text(self) -> str:
        if self.numerator == 0:
            return "zero"
        elif self.numerator == self.denominator:
            return "one whole"
        elif self.numerator == 1:
            ordinals = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth", 8: "eighth"}
            return f"one {ordinals.get(self.denominator, f'{self.denominator}th')}"
        else:
            ordinals = {2: "halves", 3: "thirds", 4: "quarters", 5: "fifths", 6: "sixths", 8: "eighths"}
            return f"{self.numerator} {ordinals.get(self.denominator, f'{self.denominator}ths')}"


def build_math_concept_scene() -> VisualScene:
    return VisualScene(
        title="Fraction Bars",
        description="Interactive bars show parts of a whole to ground abstract fractions.",
    )


def get_preset_fraction(preset: Literal["half", "quarter", "third", "two-thirds"]) -> FractionBar:
    presets = {
        "half": FractionBar(1, 2, "#4a90e2"),
        "quarter": FractionBar(1, 4, "#50c878"),
        "third": FractionBar(1, 3, "#f39c12"),
        "two-thirds": FractionBar(2, 3, "#e74c3c"),
    }
    return presets.get(preset, FractionBar(1, 2, "#4a90e2"))


def compare_fractions(frac1: FractionBar, frac2: FractionBar) -> str:
    val1 = frac1.value
    val2 = frac2.value
    if val1 > val2:
        return f"{frac1.to_text()} is greater than {frac2.to_text()}"
    elif val1 < val2:
        return f"{frac1.to_text()} is less than {frac2.to_text()}"
    else:
        return f"{frac1.to_text()} is equal to {frac2.to_text()}"
