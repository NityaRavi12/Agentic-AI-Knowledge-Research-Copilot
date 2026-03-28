"""
layer2_research/schemas.py
The output object for Layer 2.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Layer2Report:
    subquestions: list[str]
    subresults: list[Any]
    report: str
    confidence: float