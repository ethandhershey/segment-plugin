import polars as pl
import polars.selectors as cs
from scipy import stats

from tqdm import tqdm

from typing import TypedDict
from datetime import timedelta, datetime
from pathlib import Path
import heapq
import json
import time

# For rules (input constraints)
class ContinuousRule(TypedDict, total=False):
    mean: float
    overall: float

class CategoricalRule(TypedDict, total=False):
    proportion: float

# For results (output values)
class ContinuousResult(TypedDict, total=False):
    mean: float
    overall: float

class CategoricalResult(TypedDict, total=False):
    proportion: float

def create_test_control_assignment(
    df: pl.DataFrame,
    segment_col_name: str,
    date_col_name: str,
    continuous_rules: dict[str, ContinuousRule],
    categorical_rules: dict[str, CategoricalRule],
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
    
    This function performs iterative random assignment of data to test and control groups,
    evaluating each assignment against statistical rules to find the best configurations.
    It saves progress periodically and can be interrupted and resumed.
    
    Args:
        df: Polars DataFrame containing the data to be assigned
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
        >>> df = pl.DataFrame({
        ...     'segment': ['A', 'B', 'C'] * 100,
        ...     'date': ['2024-01-01'] * 300,
        ...     'value': np.random.normal(100, 20, 300),
        ...     'category': np.random.choice(['X', 'Y'], 300)
        ... })
        >>> result = create_test_control_assignment(
        ...     df=df,
        ...     segment_col_name='segment',
        ...     date_col_name='date',
        ...     continuous_rules={'value': {'mean': 0.05}},
        ...     categorical_rules={'category': {'proportion': 0.1}},
        ...     n_iters=1000
        ... )
    """
    for col_name in [segment_col_name, date_col_name]:
        dtype = df.get_column(col_name).dtype
        if dtype != pl.Categorical and dtype != pl.Enum:
            raise ValueError(
                f"{col_name} column must be of type Categorical or Enum, the column is of type {dtype}"
            )

    n_control: int = int(df.height * proportion_control + 1)
    n_test: int = df.height - n_control

    df = (
        df
        .select(list(continuous_rules.keys()) + list(categorical_rules.keys()) + [segment_col_name, date_col_name])
        .sort(pl.col(segment_col_name))
        .with_row_index(name='_index')
    )

    class PassedSeed(TypedDict):
        seed: int
        continuous_values: dict[str, ContinuousResult]
        categorical_values: dict[str, CategoricalResult]

    top_seeds: list[(float, PassedSeed)] = [(float('-inf'), PassedSeed(seed=None, continuous_values=None, categorical_values=None))] * n_save
    heapq.heapify(top_seeds)

    last_save_time = time.time()
    last_backoff_time = time.time()
    
    def save_top_seeds(current_iter: int):
        save_path = Path(str(filename_save).format(current_iter))
        # Ensure parent directories exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert heap to sorted list for saving (best scores first)
        sorted_seeds = sorted(top_seeds, key=lambda x: x[0], reverse=True)
        
        # Prepare data for JSON serialization
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'iteration': current_iter,
            'total_iterations': n_iters,
            'n_save': n_save,
            'top_seeds': [
                {
                    'score': float(score),
                    'seed': seed_data['seed'],
                    'continuous_values': seed_data['continuous_values'],
                    'categorical_values': seed_data['categorical_values']
                }
                for score, seed_data in sorted_seeds
                if seed_data['seed'] is not None  # Skip placeholder entries
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

    try:
        for i in tqdm(range(seed_start, n_iters), desc='Processing seeds'):
            shuffled_df: pl.DataFrame = shuffle_into_groups(df, segment_col_name, proportion_control, seed=i)

            score = 0.0
            continuous_results: dict[str, ContinuousResult] = {}
            categorical_results: dict[str, CategoricalResult] = {}

            passed = False
            for col_name, rules in continuous_rules.items():
                mean_rule = rules.get("mean", None)
                if mean_rule is not None:
                    means_df = (
                        shuffled_df
                        .lazy()
                        .group_by(date_col_name, '_group')
                        .agg(pl.col(col_name).mean())
                        .collect()
                    )

                    p_value = get_p_value(
                        means_df.filter(pl.col('_group') == 'control').get_column(col_name),
                        means_df.filter(pl.col('_group') == 'test').get_column(col_name)
                    )

                    if p_value < mean_rule:
                        break

                    continuous_results.setdefault(col_name, {})['mean'] = p_value
                    score += (p_value - mean_rule) / (1 - mean_rule)

                overall_rule = rules.get("overall", None)
                if overall_rule is not None:
                    p_value = get_p_value(
                        shuffled_df.filter(pl.col('_group') == 'control').get_column(col_name),
                        shuffled_df.filter(pl.col('_group') == 'test').get_column(col_name)
                    )

                    if p_value < overall_rule:
                        break

                    continuous_results.setdefault(col_name, {})['overall'] = p_value
                    score += (p_value - overall_rule) / (1 - overall_rule)
            else:
                passed = True
            if not passed:
                continue
            passed = False

            for col_name, rules in categorical_rules.items():
                proportion_rule = rules.get("proportion", None)
                if proportion_rule is not None:
                    max_difference = (
                        shuffled_df
                        .lazy()
                        .group_by('_group', col_name)
                        .agg(pl.count().alias('_count'))
                        .with_columns(
                            (pl.col('_count') / pl.col('_count').sum().over('_group')).alias('_proportion')
                        )
                        .collect()
                        .pivot(values='_proportion', index=col_name, on='_group')
                        .select(
                            (pl.col('test') - pl.col('control')).max()
                        )
                        .item()
                    )
                    
                    if max_difference > proportion_rule:
                        break

                    categorical_results.setdefault(col_name, {})['proportion'] = max_difference
            else:
                passed = True
            if not passed:
                continue
            passed = False

            entry = (
                score,
                PassedSeed(
                    seed=i,
                    continuous_values=continuous_results,
                    categorical_values=categorical_results,
                )
            )
            if entry != heapq.heappushpop(top_seeds, entry):
                # TODO: added_seed = True
                pass

            current_time = time.time()

            is_time = lambda interval, last_time: (
                (isinstance(interval, int) and (i+1) % interval == 0)
                or (isinstance(interval, timedelta) and current_time - last_time >= interval.total_seconds())
            )

            if is_time(interval_save, last_save_time):
                save_top_seeds(i)
                last_save_time = current_time
            
            if is_time(interval_backoff, last_backoff_time):
                # TODO: Add backoff, but backoff might not even be needed
                pass
    except KeyboardInterrupt:
        print(f"\nInterrupted at iteration {i+1}.")

    # Final save at the end
    save_top_seeds(i+1)
    
    return top_seeds

def get_p_value(group_1: pl.Series, group_2: pl.Series) -> float:
    _statistic, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)
    return p_value

def shuffle_into_groups(
    df: pl.DataFrame,
    segment_col_name: str,
    proportion_control: float,
    seed: int,
    group_col_name: str = '_group'
) -> pl.DataFrame:
    """
    Shuffle data into test and control groups while maintaining segment balance.
    
    This function randomly assigns rows to test and control groups while ensuring
    that the specified proportion of data goes to the control group. The assignment
    is done within each segment to maintain balance across different segments.
    
    Args:
        df: Polars DataFrame containing the data to be shuffled
        segment_col_name: Name of the column containing segment identifiers
        proportion_control: Proportion of data to assign to control group (0.0 to 1.0)
        seed: Random seed for reproducible shuffling
        group_col_name: Name for the new column that will contain group assignments
    
    Returns:
        Polars DataFrame with an additional column containing group assignments.
        The group column will contain 'control' and 'test' values.
    
    Example:
        >>> df = pl.DataFrame({
        ...     'segment': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...     'value': [1, 2, 3, 4, 5, 6]
        ... })
        >>> result = shuffle_into_groups(df, 'segment', 0.5, seed=42)
        >>> print(result)
        # Returns DataFrame with '_group' column containing 'control' and 'test'
    """
    # 50% 0 and 50% 1, although this doesn't have to be the case.
    # You could make this account for the small groups and perfectly balance based
    # on the input ratio but I think this is fine.
    zero_or_one: pl.Series = pl.concat([
        pl.repeat(0, int(df.height / 2 + 0.5)),
        pl.repeat(1, int(df.height / 2))
    ])

    shuffled_df = (
        df
        .sample(fraction=1, shuffle=True, seed=seed)
        .lazy()
        .with_columns(
            zero_or_one.alias('_zero_or_one'),
            pl.int_range(pl.len()).over(segment_col_name).alias('_group_index'),
            pl.len().over(segment_col_name).alias('_group_len')
        )
        .with_columns(
            pl.when(pl.col('_group_index') < (proportion_control * pl.col('_group_len') - pl.col('_zero_or_one')))
            .then(pl.lit("control"))
            .otherwise(pl.lit("test"))
            .cast(pl.Categorical)
            .alias("_group")
        )
        .drop(cs.starts_with('_').exclude('_group'))
        .collect()
    )

    return shuffled_df

def create_segment_column(
    df: pl.DataFrame,
    segment_cols: dict[str, int],
    segment_col_name: str = 'segments',
) -> pl.Series:
    """
    Create a segment column by combining multiple categorical columns.
    
    This function creates a new segment column by quantizing continuous columns
    and combining them with existing categorical columns. Each column is divided
    into the specified number of segments, and the results are concatenated to
    create unique segment identifiers.
    
    Args:
        df: Polars DataFrame containing the data
        segment_cols: Dictionary mapping column names to number of segments.
            For continuous columns, this specifies how many quantiles to create.
            For categorical columns, this specifies how many categories to use.
        segment_col_name: Name for the new segment column
    
    Returns:
        Polars Series containing the segment assignments as categorical values.
        Each value is a string representing the combined segment identifier.
    
    Example:
        >>> df = pl.DataFrame({
        ...     'age': [25, 30, 35, 40, 45],
        ...     'income': [50000, 60000, 70000, 80000, 90000],
        ...     'region': ['North', 'South', 'East', 'West', 'Central']
        ... })
        >>> segments = create_segment_column(
        ...     df=df,
        ...     segment_cols={'age': 3, 'income': 2, 'region': 2},
        ...     segment_col_name='customer_segments'
        ... )
        >>> print(segments)
        # Returns Series with values like '0_0_0', '1_1_1', etc.
    """
    segment_col = pl.repeat('', df.height, eager=True)

    for col_name, n_segments in segment_cols.items():
        segment_col += df.get_column(col_name).qcut(n_segments, labels=[f"{i}" for i in range(n_segments)])
    
    segment_col = segment_col.cast(pl.Categorical).alias(segment_col_name)
    
    return segment_col