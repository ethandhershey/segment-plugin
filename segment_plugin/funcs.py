from typing import TypeVar
import polars as pl
import re

def to_snake_case(string: str):
    abbreviations = {
        'TRx': 'Trx',
        'NRx': 'Nrx',
        'compasid': 'compas_id',
    }
    
    s0 = string
    for abbrev, replacement in abbreviations.items():
        s0 = s0.replace(abbrev, replacement)
    
    # Insert underscore before capital letters that follow lowercase/digits
    s1 = re.sub('([a-z0-9])([A-Z])', r'\1_\2', s0)
    # Handle consecutive capitals followed by lowercase
    s2 = re.sub('([A-Z])([A-Z][a-z])', r'\1_\2', s1)
    # Replace spaces and hyphens, then lowercase
    return s2.lower().replace(' ', '_').replace('-', '_')

def describe(col: str, prefix: str | None = None):
    if prefix is None:
        prefix = col + "_"
    
    return [
        pl.col(col).count().alias(prefix + 'count'),
        pl.col(col).mean().alias(prefix + 'mean'),
        pl.col(col).std().alias(prefix + 'std'),
        pl.col(col).min().alias(prefix + 'min'),
        pl.col(col).quantile(0.25).alias(prefix + '25%'),
        pl.col(col).quantile(0.5).alias(prefix + '50%'),
        pl.col(col).quantile(0.75).alias(prefix + '75%'),
        pl.col(col).max().alias(prefix + 'max'),
    ]

PolarsFrame = TypeVar('PolarsFrame', pl.DataFrame, pl.LazyFrame)

# You can also do this: .pipe(lambda df: df.write_clipboard() or df)
# but this function is cleaner and handles collecting lazyframes
def df_write_clipboard(df: PolarsFrame) -> PolarsFrame:
    if isinstance(df, pl.LazyFrame):
        df.collect().write_clipboard()
    else:
        df.write_clipboard()
    return df

# def df_display(df: PolarsFrame) -> PolarsFrame:
#     if isinstance(df, pl.LazyFrame):
#         display(df.collect())
#     else:
#         display(df)
#     return df