"""
Pandas wrappers for segment_plugin functions.

This submodule provides pandas-compatible versions of the main segment_plugin functions.
All functions automatically convert pandas DataFrames/Series to polars, call the original
function, and convert the result back to pandas.

Example usage:
    import segment_plugin.pandas as sp_pd
    
    # Use pandas wrappers
    result = sp_pd.create_test_control_assignment_pd(df, ...)
    segments = sp_pd.create_segment_column_pd(df, ...)
"""

from segment_plugin.segment_pd import (
    create_test_control_assignment_pd,
    create_segment_column_pd,
    shuffle_into_groups_pd,
) 