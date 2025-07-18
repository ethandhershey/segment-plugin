from __future__ import annotations

# import the contents of the Rust library into the Python extension
from segment_plugin import *

# optional: include the documentation from the Rust module
from segment_plugin import __doc__  # noqa: F401

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from segment_plugin._internal import __version__ as __version__
import segment_plugin as sp

if TYPE_CHECKING:
    from segment_plugin.typing import IntoExprColumn

LIB = Path(__file__).parent

# Functions

def noop(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="noop",
        is_elementwise=True,
    )

def abs_i64(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="abs_i64",
        is_elementwise=True,
    )

def cum_sum(expr: IntoExprColumn) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="cum_sum",
        is_elementwise=False,
    )

def guess_the_number() -> None:
    sp._internal.guess_the_number()

from segment_plugin.funcs import to_snake_case
from segment_plugin.segment import create_segment_column, create_test_control_assignment
from segment_plugin.segment_pd import create_test_control_assignment_pd