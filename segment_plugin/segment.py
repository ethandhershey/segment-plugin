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
import math

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
    id_col_name: str,
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
    Iteratively search for optimal test/control assignments for A/B testing, subject to statistical constraints.

    This function randomly assigns data to test and control groups, evaluating each assignment against user-specified statistical rules. It periodically saves progress and supports interruption/resumption. The top scoring seeds (assignments) are tracked and returned.

    Args:
        df (pl.DataFrame): Data to be assigned.
        id_col_name (str): Name of the column with unique identifiers.
        segment_col_name (str): Name of the column with segment identifiers.
        date_col_name (str): Name of the column with date/time information.
        continuous_rules (dict[str, ContinuousRule]): Rules for continuous columns.
        categorical_rules (dict[str, CategoricalRule]): Rules for categorical columns.
        proportion_control (float, optional): Proportion of data to assign to control group. Defaults to 0.1.
        percent_backoff (float, optional): Percentage for relaxing rules over time. Defaults to 0.1.
        interval_backoff (timedelta|int, optional): Time or iteration interval for relaxing rules. Defaults to 2 hours.
        interval_save (timedelta|int, optional): Time or iteration interval for saving progress. Defaults to 1 hour.
        filename_save (Path, optional): File path pattern for saving results. Defaults to 'tc_saves/tc_save_{}.json'.
        seed_start (int, optional): Starting random seed. Defaults to 0.
        n_iters (int, optional): Maximum number of iterations. Defaults to 1e9.
        n_save (int, optional): Number of top seeds to save. Defaults to 100.

    Returns:
        list: List of (score, seed_data) tuples for the top assignments. Each seed_data contains:
            - seed (int): The random seed used.
            - continuous_values (dict): p-values for continuous variables.
            - categorical_values (dict): proportion differences for categorical variables.

    Raises:
        ValueError: If segment_col_name or date_col_name columns are not Categorical or Enum type.

    Example:
        >>> df = pl.DataFrame({
        ...     'segment': ['A', 'B', 'C'] * 100,
        ...     'date': ['2024-01-01'] * 300,
        ...     'value': np.random.normal(100, 20, 300),
        ...     'category': np.random.choice(['X', 'Y'], 300)
        ... })
        >>> result = create_test_control_assignment(
        ...     df=df,
        ...     id_col_name='id',
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
        .select(list(continuous_rules.keys()) + list(categorical_rules.keys()) + [id_col_name, segment_col_name, date_col_name])
        .sort(pl.col(segment_col_name))
        .with_row_index(name='_index')
    )

    grouped_df = (
        df
        .group_by(id_col_name, maintain_order=True)
        .agg(pl.col(segment_col_name).str.join())
    )

    print(grouped_df.get_column(segment_col_name).value_counts())

    class PassedSeed(TypedDict):
        seed: int
        continuous_values: dict[str, ContinuousResult]
        categorical_values: dict[str, CategoricalResult]

    top_seeds: list[(float, PassedSeed)] = [(float('-inf'), PassedSeed(seed=None, continuous_values=None, categorical_values=None))] * n_save
    heapq.heapify(top_seeds)

    added_seeds = 0

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
            mapped_ids: pl.DataFrame = assign_groups_by_segments(
                grouped_df=grouped_df,
                id_col_name=id_col_name,
                segment_col_name=segment_col_name,
                proportion_control=proportion_control,
                seed=i,
                group_col_name='_group',
            )

            df_with_group: pl.DataFrame = df.join(mapped_ids, on=id_col_name, how='left')

            score = 0.0
            continuous_results: dict[str, ContinuousResult] = {}
            categorical_results: dict[str, CategoricalResult] = {}

            passed = False
            for col_name, rules in continuous_rules.items():
                mean_rule = rules.get("mean", None)
                if mean_rule is not None:
                    means_df = (
                        df_with_group
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
                        df_with_group.filter(pl.col('_group') == 'control').get_column(col_name),
                        df_with_group.filter(pl.col('_group') == 'test').get_column(col_name)
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
                        df_with_group
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
                added_seeds += 1
                pass

            if math.log(added_seeds, 2).is_integer():
                print(f'Total seeds: {added_seeds}')

            current_time = time.time()

            is_time = lambda interval, last_time: (
                (isinstance(interval, int) and (i+1) % interval == 0)
                or (isinstance(interval, timedelta) and current_time - last_time >= interval.total_seconds())
            )

            if is_time(interval_save, last_save_time):
                save_top_seeds(i)
                last_save_time = current_time
            
            if is_time(interval_backoff, last_backoff_time):
                for col_name, rules in continuous_rules.items():
                    for key in list(rules.keys()):
                        if rules[key] is not None:
                            rules[key] = rules[key] * (1 + percent_backoff)
                for col_name, rules in categorical_rules.items():
                    for key in list(rules.keys()):
                        if rules[key] is not None:
                            rules[key] = rules[key] * (1 + percent_backoff)
                last_backoff_time = current_time
                print(f"Backoff applied at iteration {i+1}")
    except KeyboardInterrupt:
        print(f"\nInterrupted at iteration {i+1}")

    # Final save at the end
    save_top_seeds(i+1)
    
    return top_seeds

def get_p_value(group_1: pl.Series, group_2: pl.Series) -> float:
    _statistic, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)
    return p_value

def assign_groups_by_segments(
    id_col_name: str,
    segment_col_name: str,
    proportion_control: float,
    seed: int,
    group_col_name: str = 'group',
    df: pl.DataFrame | None = None,
    grouped_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    Randomly assign rows to test and control groups, maintaining segment balance.

    This function shuffles and assigns rows to test/control groups, ensuring the specified proportion for control and balancing assignments within each segment.

    Args:
        id_col_name (str): Name of the column with unique identifiers.
        segment_col_name (str): Name of the column with segment identifiers.
        proportion_control (float): Proportion of data to assign to control group (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
        group_col_name (str, optional): Name for the new group assignment column. Defaults to 'group'.
        df (pl.DataFrame, optional): Data to be assigned. Required if grouped_df is not provided.
        grouped_df (pl.DataFrame, optional): Pre-grouped data. Required if df is not provided.

    Returns:
        pl.DataFrame: DataFrame with an additional column for group assignments ('control' or 'test').

    Raises:
        ValueError: If neither or both of df and grouped_df are provided.

    Example:
        >>> df = pl.DataFrame({
        ...     'id': [1, 2, 3, 4, 5, 6],
        ...     'segment': ['A', 'A', 'B', 'B', 'C', 'C'],
        ...     'value': [1, 2, 3, 4, 5, 6]
        ... })
        >>> result = assign_groups_by_segments(df=df, id_col_name='id', segment_col_name='segment', proportion_control=0.5, seed=42)
        >>> print(result)
        # Returns DataFrame with 'group' column containing 'control' and 'test'
    """
    if grouped_df is not None:
        pass
    elif df is not None:
        grouped_df = (
            df
            .group_by(id_col_name, maintain_order=True)
            .agg(pl.col(segment_col_name).str.join())
        )
    else:
        raise ValueError(f"One and only one of df or grouped_df must be not be None")

    # 50% 0 and 50% 1, although this doesn't have to be the case.
    # You could make this account for the small groups and perfectly balance based
    # on the input ratio but I think this is fine.
    zero_or_one: pl.Series = pl.concat([
        pl.repeat(0, int(grouped_df.height / 2 + 0.5)),
        pl.repeat(1, int(grouped_df.height / 2))
    ])

    shuffled_df = (
        grouped_df
        .sample(fraction=1, shuffle=True, seed=seed)
        .lazy()
        .select(pl.col(id_col_name), pl.col(segment_col_name))
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
            .alias(group_col_name),
        )
        .drop(cs.starts_with('_').exclude(group_col_name))
        .collect()
    )

    return shuffled_df

def create_segment_column(
    segment_cols: dict[str, int],
    df: pl.DataFrame | None = None,
    df_id_col_name: str | None = None,
    grouped_df: pl.DataFrame | None = None,
    segment_col_name: str = 'segments',
    show: bool = True,
) -> pl.Series:
    """
    Create a segment column by combining and quantizing multiple columns.

    This function creates a new segment column by quantizing continuous columns and combining them with categorical columns. Each column is divided into the specified number of segments, and the results are concatenated to create unique segment identifiers.

    Args:
        segment_cols (dict[str, int]): Mapping of column names to number of segments (quantiles or categories).
        df (pl.DataFrame, optional): Data to use. Required if grouped_df is not provided.
        df_id_col_name (str, optional): Name of the ID column in df. Required if df is provided.
        grouped_df (pl.DataFrame, optional): Pre-grouped data. Required if df is not provided.
        segment_col_name (str, optional): Name for the new segment column. Defaults to 'segments'.
        show (bool, optional): Whether to print value counts for the new segment column. Defaults to True.

    Returns:
        pl.Series: Series containing the segment assignments as categorical values.

    Raises:
        ValueError: If neither or both of df and grouped_df are provided, or if n_segments < 2 for any column.

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
        # Returns Series with values like '010', '111', etc.
    """
    if grouped_df is not None and df is None:
        pass
    elif df is not None and grouped_df is None:
        if df_id_col_name is None:
            raise ValueError(f"df_id_col_name must be set along with df")
        grouped_df = (
            df
            .group_by(df_id_col_name)
            .agg(pl.col(segment_cols.keys()).sum())
        )
    else:
        raise ValueError(f"One and only one of df or grouped_df must be not be None")

    segment_col = pl.repeat('', grouped_df.height, eager=True)

    for col_name, n_segments in segment_cols.items():
        if n_segments < 2:
            raise ValueError(f'{n_segments} is less then 2, n_segments must be at least 2')
        try:
            segment_col += grouped_df.get_column(col_name).qcut(n_segments, labels=[f"{i}" for i in range(n_segments)])
        except pl.exceptions.DuplicateError:
            raise ValueError(f'Column \"{col_name}\" could not be split into {n_segments}, try lowering the number of segments')
    
    
    segment_col = segment_col.cast(pl.Categorical).alias(segment_col_name)

    if show:
        print(segment_col.value_counts())

    if df is None:
        return segment_col

    final_column = (
        df
        .join(
            grouped_df.with_columns(segment_col),
            on=df_id_col_name,
            how='left'
        )
        .get_column(segment_col_name)
    )
    
    return final_column