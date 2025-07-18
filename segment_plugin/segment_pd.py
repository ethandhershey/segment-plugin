from segment_plugin.segment import create_segment_column, create_test_control_assignment

import pandas as pd
import polars as pl

def create_test_control_assignment_pd(df=pd.DataFrame, **kwargs):
    df: pl.DataFrame = pl.from_pandas(df)
    df.with_columns(pl.col(kwargs['date_col_name']).map_elements(lambda x: str(x), return_dtype=pl.Utf8).cast(pl.Categorical))

    kwargs['df'] = df
    return create_test_control_assignment(**kwargs)