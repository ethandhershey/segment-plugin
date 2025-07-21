import polars as pl
import segment_plugin as sp

df = pl.DataFrame({
    'a': [1, -1, None],
    'b': [4.1, 5.2, -6.3],
    'c': ['hello', 'everybody!', '!']
})
print(df.with_columns(sp.cum_sum('a').name.suffix('_cum_sum')))

# sp.guess_the_number()
print(sp.to_snake_case('whatTTa'))

df = pl.DataFrame({
    "a": [23.5, 24.1, 22.8, 25.3, 23.9],
    "b": [45.2, 48.7, 44.3, 50.1, 46.8],
    "c": [1013.5, 1012.8, 1014.1, 1011.9, 1013.5],
    "d": [5.4, 6.2, 4.8, 7.1, 5.9]
})
print(
    sp.create_segment_column(df=df, segment_cols={'b': 2, 'c': 2})
)

import polars as pl
import segment_plugin as sp
import numpy as np
from datetime import timedelta, datetime

# Create test DataFrame
np.random.seed(42)
n_rows = 10000

# Generate month-level dates (Jan-Dec 2024)
months = [datetime(2024, month, 1) for month in range(1, 13)]  # Jan to Dec
monthly_dates = np.random.choice(months, n_rows)

# Generate test data
df = pl.DataFrame({
    # Date column (month level)
    "date": monthly_dates,
    
    # Segment column
    "customer_segment": np.random.choice(["A", "B", "C", "D"], n_rows),
    
    # Continuous columns
    "age": np.random.normal(40, 15, n_rows),
    "income": np.random.lognormal(10.5, 0.5, n_rows),  # Log-normal for income
    "purchase_amount": np.random.exponential(100, n_rows),
    "days_since_signup": np.random.uniform(0, 365, n_rows),
    
    # Categorical columns
    "gender": np.random.choice(["M", "F", "Other"], n_rows, p=[0.45, 0.45, 0.1]),
    "region": np.random.choice(["North", "South", "East", "West"], n_rows),
    "membership_type": np.random.choice(["Basic", "Premium", "Gold"], n_rows, p=[0.6, 0.3, 0.1]),
    
    # Additional columns
    "customer_id": range(n_rows),
    "is_active": np.random.choice([True, False], n_rows, p=[0.8, 0.2])
})

df = df.with_columns(
    pl.col('date').map_elements(lambda x: str(x.month), return_dtype=pl.Utf8).cast(pl.Categorical),
    pl.col('customer_segment').cast(pl.Categorical)
)

print(df.get_column('date').value_counts())

# Define continuous columns options
continuous_cols = {
    "age": {
        "mean": 0.5,
        "overall": 0.1
    },
    "income": {
        "mean": 0.5
    },
    "purchase_amount": {
        "mean": 0.5,
    },
    "days_since_signup": {
        "overall": 0.1
    },
}

# Define categorical columns options
categorical_cols = {
    "gender": {
        "proportion": 0.05
    },
    "region": {
        "proportion": 0.03
    },
    "membership_type": {
        "proportion": 0.01
    }
}

print(df.get_column('customer_segment').value_counts())

# Call the function with test inputs
result = sp.create_test_control_assignment(
    df=df,
    id_col_name='customer_id',
    segment_col_name="customer_segment",
    date_col_name="date",
    continuous_rules=continuous_cols,
    categorical_rules=categorical_cols,
    proportion_control=0.2,
    percent_backoff=0.05,
    interval_save=timedelta(seconds=3),
    n_iters=1000
)

print(result)