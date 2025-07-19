from segment_plugin.segment import (
    create_segment_column, 
    create_test_control_assignment, 
    shuffle_into_groups,
)

import pandas as pd
import polars as pl
from typing import Any
from pathlib import Path
from datetime import timedelta

def create_test_control_assignment_pd(
    df: pd.DataFrame,
    segment_col_name: str,
    date_col_name: str,
    continuous_rules: dict[str, Any],
    categorical_rules: dict[str, Any],
    proportion_control: float = 0.1,
    percent_backoff: float = 0.1,
    interval_backoff: timedelta | int = timedelta(hours=2),
    interval_save: timedelta | int = timedelta(hours=1),
    filename_save: Path = Path('tc_saves/tc_save_{}.json'),
    seed_start: int = 0,
    n_iters: int = int(1e9),
    n_save: int = 100,
) -> list:
    """
    Create optimal test/control assignments for A/B testing with statistical constraints.
    
    Pandas wrapper for the polars create_test_control_assignment function. This function
    automatically converts pandas DataFrames to polars, calls the underlying function,
    and returns the same results. It performs iterative random assignment of data to
    test and control groups, evaluating each assignment against statistical rules to
    find the best configurations.
    
    Args:
        df: pandas DataFrame containing the data to be assigned
        segment_col_name: Name of the column containing segment identifiers
        date_col_name: Name of the column containing date/time information
        continuous_rules: Dictionary mapping continuous column names to statistical rules.
            Each rule can specify:
            - 'mean': Maximum p-value threshold for mean differences between groups
            - 'overall': Maximum p-value threshold for overall differences between groups
        categorical_rules: Dictionary mapping categorical column names to statistical rules.
            Each rule can specify:
            - 'proportion': Maximum proportion difference allowed between groups
        proportion_control: Proportion of data to assign to control group (default: 0.1)
        percent_backoff: Percentage for backoff mechanism (default: 0.1)
        interval_backoff: Time interval or iteration count for backoff (default: 2 hours)
        interval_save: Time interval or iteration count for saving progress (default: 1 hour)
        filename_save: File path pattern for saving results, uses {} for iteration number
        seed_start: Starting random seed for iteration (default: 0)
        n_iters: Maximum number of iterations to perform (default: 1 billion)
        n_save: Number of top seeds to save (default: 100)
    
    Returns:
        List of tuples containing (score, seed_data) for the top performing assignments.
        Each seed_data contains:
        - seed: The random seed that produced this assignment
        - continuous_values: Dictionary of p-values for continuous variables
        - categorical_values: Dictionary of proportion differences for categorical variables
    
    Raises:
        ValueError: If segment_col_name or date_col_name columns are not Categorical or Enum type
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({
        ...     'segment': ['A', 'B', 'C'] * 100,
        ...     'date': ['2024-01-01'] * 300,
        ...     'value': np.random.normal(100, 20, 300),
        ...     'category': np.random.choice(['X', 'Y'], 300)
        ... })
        >>> result = create_test_control_assignment_pd(
        ...     df=df,
        ...     segment_col_name='segment',
        ...     date_col_name='date',
        ...     continuous_rules={'value': {'mean': 0.05}},
        ...     categorical_rules={'category': {'proportion': 0.1}},
        ...     n_iters=1000
        ... )
    """
    # Convert pandas DataFrame to polars
    pl_df = pl.from_pandas(df)
    
    # Convert date column to categorical if it's not already
    if pl_df.get_column(date_col_name).dtype not in [pl.Categorical, pl.Enum]:
        pl_df = pl_df.with_columns(
            pl.col(date_col_name).cast(pl.Utf8).cast(pl.Categorical)
        )
    
    return create_test_control_assignment(
        df=pl_df,
        segment_col_name=segment_col_name,
        date_col_name=date_col_name,
        continuous_rules=continuous_rules,
        categorical_rules=categorical_rules,
        proportion_control=proportion_control,
        percent_backoff=percent_backoff,
        interval_backoff=interval_backoff,
        interval_save=interval_save,
        filename_save=filename_save,
        seed_start=seed_start,
        n_iters=n_iters,
        n_save=n_save,
    )

def create_segment_column_pd(
    df: pd.DataFrame,
    segment_cols: dict[str, int],
    segment_col_name: str = 'segments',
) -> pd.Series:
    """
    Create a segment column by combining multiple categorical columns.
    
    Pandas wrapper for the polars create_segment_column function. This function
    automatically converts pandas DataFrames to polars, calls the underlying function,
    and returns the results as a pandas Series. It creates a new segment column by
    quantizing continuous columns and combining them with existing categorical columns.
    
    Args:
        df: pandas DataFrame containing the data
        segment_cols: Dictionary mapping column names to number of segments.
            For continuous columns, this specifies how many quantiles to create.
            For categorical columns, this specifies how many categories to use.
        segment_col_name: Name for the new segment column
    
    Returns:
        pandas Series containing the segment assignments as categorical values.
        Each value is a string representing the combined segment identifier.
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'age': [25, 30, 35, 40, 45],
        ...     'income': [50000, 60000, 70000, 80000, 90000],
        ...     'region': ['North', 'South', 'East', 'West', 'Central']
        ... })
        >>> segments = create_segment_column_pd(
        ...     df=df,
        ...     segment_cols={'age': 3, 'income': 2, 'region': 2},
        ...     segment_col_name='customer_segments'
        ... )
        >>> print(segments)
        # Returns pandas Series with values like '0_0_0', '1_1_1', etc.
    """
    # Convert pandas DataFrame to polars
    pl_df = pl.from_pandas(df)
    
    # Create segment column using polars function
    pl_segment_col = create_segment_column(
        df=pl_df,
        segment_cols=segment_cols,
        segment_col_name=segment_col_name,
    )
    
    # Convert back to pandas Series
    return pl_segment_col.to_pandas()

def shuffle_into_groups_pd(
    df: pd.DataFrame,
    segment_col_name: str,
    proportion_control: float,
    seed: int,
    group_col_name: str = '_group'
) -> pd.DataFrame:
    """
    Shuffle data into test and control groups while maintaining segment balance.
    
    Pandas wrapper for the polars shuffle_into_groups function. This function
    automatically converts pandas DataFrames to polars, calls the underlying function,
    and returns the results as a pandas DataFrame. It randomly assigns rows to test
    and control groups while ensuring that the specified proportion of data goes to
    the control group.
    
    Args:
        df: pandas DataFrame containing the data to be shuffled
        segment_col_name: Name of the column containing segment identifiers
        proportion_control: Proportion of data to assign to control group (0.0 to 1.0)
        seed: Random seed for reproducible shuffling
        group_col_name: Name for the new column that will contain group assignments
    
    Returns:
        pandas DataFrame with an additional column containing group assignments.
        The group column will contain 'control' and 'test' values.
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'segment': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...     'value': [1, 2, 3, 4, 5, 6]
        ... })
        >>> result = shuffle_into_groups_pd(df, 'segment', 0.5, seed=42)
        >>> print(result)
        # Returns pandas DataFrame with '_group' column containing 'control' and 'test'
    """
    # Convert pandas DataFrame to polars
    pl_df = pl.from_pandas(df)
    
    # Apply shuffle_into_groups using polars function
    pl_result = shuffle_into_groups(
        df=pl_df,
        segment_col_name=segment_col_name,
        proportion_control=proportion_control,
        seed=seed,
        group_col_name=group_col_name
    )
    
    # Convert back to pandas DataFrame
    return pl_result.to_pandas()