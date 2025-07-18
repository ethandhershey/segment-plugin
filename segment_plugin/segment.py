import polars as pl
import polars.selectors as cs
from scipy import stats

from tqdm import tqdm

from typing import Iterable, Dict, NamedTuple, Optional, TypedDict, List
from datetime import timedelta
from pathlib import Path
import heapq

class ContinuousValues(TypedDict, total=False):
    mean: float
    overall: float

class CategoricalValues(TypedDict, total=False):
    proportion: float

def create_test_control_assignment(
    df: pl.DataFrame,
    segment_col_name: str,
    date_col_name: str,
    continuous_rules: Dict[str, ContinuousValues],
    categorical_rules: Dict[str, CategoricalValues],
    fraction_control: float = 0.1,
    percent_backoff: int = 10,
    interval_save: timedelta = timedelta(hours=1),
    filename_save: Path = Path('tc_save_{}.json'),
    seed_start: int = 0,
    n_iters: int = int(1e9),
    n_save: int = 100,
) -> List:
    if df.get_column(date_col_name).dtype != pl.Categorical and df.get_column(date_col_name).dtype != pl.Enum:
        raise ValueError(
            f'date_col_name column must be of type Categorical or Enum, the column is of type {df.get_column(date_col_name).dtype}'
        )

    n_control: int = int(df.height * fraction_control + 1)
    n_test: int = df.height - n_control

    df = (
        df
        .select(list(continuous_rules.keys()) + list(categorical_rules.keys()) + [segment_col_name, date_col_name])
        .sort(pl.col(segment_col_name))
        .with_row_index(name='_index')
    )

    class PassedSeed(TypedDict):
        seed: int
        continuous_values: ContinuousValues
        categorical_values: CategoricalValues

    top_seeds: List[(float, PassedSeed)] = [(float('-inf'), PassedSeed(seed=None, continuous_values=None, categorical_values=None))] * n_save
    heapq.heapify(top_seeds)

    for i in tqdm(range(n_iters), desc='Processing seeds'):
        zero_or_one: pl.Series = pl.concat([
            pl.repeat(0, int(df.height / 2 + 0.5)),
            pl.repeat(1, int(df.height / 2))
        ])

        shuffled_df = (
            df
            .sample(fraction=1, shuffle=True, seed=i)
            .lazy()
            .with_columns(
                zero_or_one.alias('_zero_or_one'),
                pl.int_range(pl.len()).over(segment_col_name).alias('_group_index'),
                pl.len().over(segment_col_name).alias('_group_len')
            )
            .with_columns(
                pl.when(pl.col('_group_index') < (fraction_control * pl.col('_group_len') - pl.col('_zero_or_one')))
                .then(pl.lit("control"))
                .otherwise(pl.lit("test"))
                .cast(pl.Categorical)
                .alias("_group")
            )
            .drop(cs.starts_with('_').exclude('_group'))
            .collect()
        )

        # print(shuffled_df.group_by('customer_segment').agg(pl.col('_assignment').value_counts()))

        score = 0
        continuous_values: ContinuousValues = {}
        categorical_values: CategoricalValues = {}

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

                continuous_values['mean'] = p_value
                score += (p_value - mean_rule) / (1 - mean_rule)

            overall_rule = rules.get("overall", None)
            if overall_rule is not None:
                p_value = get_p_value(
                    shuffled_df.filter(pl.col('_group') == 'control').get_column(col_name),
                    shuffled_df.filter(pl.col('_group') == 'test').get_column(col_name)
                )

                if p_value < overall_rule:
                    break

                continuous_values['overall'] = p_value
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
                        (pl.col('test') - pl.col('control') > proportion_rule).max()
                    )
                    .item()
                )
                
                if max_difference > proportion_rule:
                    break

                categorical_values['proportion'] = max_difference
        else:
            passed = True
        if not passed:
            continue
        passed = False

        heapq.heappushpop(top_seeds, (
            score,
            PassedSeed(
                seed=i,
                continuous_values=continuous_values,
                categorical_values=categorical_rules,
            )
        ))

    return top_seeds

def get_p_value(group_1: pl.Series, group_2: pl.Series):
    _statistic, p_value = stats.ttest_ind(group_1, group_2, equal_var=False)

    return p_value


def create_segment_column(
    df: pl.DataFrame,
    segment_cols: Dict[str, int],
    segment_col_name: str = 'combined_segments',
) -> pl.Series:
    segment_col = pl.repeat('', df.height, eager=True)

    for col_name, n_segments in segment_cols.items():
        segment_col += df.get_column(col_name).qcut(n_segments, labels=[f"{i}" for i in range(n_segments)])
    
    segment_col = segment_col.cast(pl.Categorical).alias(segment_col_name)
    
    return segment_col