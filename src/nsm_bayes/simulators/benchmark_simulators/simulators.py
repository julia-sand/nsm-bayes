"""Benchmark simulators used in the experiments.

Re-exports :class:`GandKSimulator` (also available as ``GNKSimulator``),
:class:`SIRSimulator` and :class:`TurinSimulator`.
"""

from .gandk import GandKSimulator
from .sir import SIRSimulator
from .turin import TurinSimulator

__all__ = [
    "SIRSimulator",
    "TurinSimulator",
    "GandKSimulator",
    "GNKSimulator",
]

GNKSimulator = GandKSimulator

