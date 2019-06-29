# Copyright 2018 QuantRocket LLC - All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
from quantrocket.moonshot import read_moonshot_csv
from moonchart.utils import intraday_to_daily
from .tears import create_full_tear_sheet
from .quantrocket_utils import pad_initial

def _get_benchmark_returns(benchmark_prices):
    """
    Returns a Series of benchmark prices, if any. If more than one column has
    benchmark prices, uses the first.
    """
    have_benchmarks = benchmark_prices.notnull().any(axis=0)
    have_benchmarks = have_benchmarks[have_benchmarks]
    if have_benchmarks.empty:
        return None

    col = have_benchmarks.index[0]
    if len(have_benchmarks.index) > 1:
        import warnings
        warnings.warn("Multiple benchmarks found, only using first ({0})".format(col))

    benchmark_prices = benchmark_prices[col]
    benchmark_prices.name = "benchmark"
    return benchmark_prices.pct_change()

def from_moonshot(results, **kwargs):
    """
    Creates a full tear sheet from a moonshot backtest results DataFrame.

    Additional kwargs are passed to create_full_tear_sheet.

    Parameters
    ----------
    results : DataFrame
        multiindex (Field, Date) DataFrame of backtest results

    Returns
    -------
    None
    """
    if "Time" in results.index.names:
        results = intraday_to_daily(results)

    # pandas DatetimeIndexes are serialized with UTC offsets, and pandas
    # parses them back to UTC but doesn't set the tz; pyfolio needs tz-aware
    if not results.index.get_level_values("Date").tz:
        results = results.tz_localize("UTC", level="Date")

    returns = results.loc["Return"].sum(axis=1)
    positions = results.loc["NetExposure"]
    positions["cash"] = 1 - positions.sum(axis=1)

    returns.name = "returns"
    returns = pad_initial(returns)

    fields = results.index.get_level_values("Field").unique()
    if "Benchmark" in fields:
        benchmark_rets = _get_benchmark_returns(
            results.loc["Benchmark"].astype(np.float64))
        benchmark_rets.name = "benchmark_returns"
        benchmark_rets = pad_initial(benchmark_rets)
        kwargs["benchmark_rets"] = benchmark_rets

    return create_full_tear_sheet(
        returns,
        positions=positions,
        **kwargs
    )

def from_moonshot_csv(filepath_or_buffer, **kwargs):
    """
    Creates a full tear sheet from a moonshot backtest results CSV.

    Additional kwargs are passed to create_full_tear_sheet.

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        filepath or file-like object of the CSV

    Returns
    -------
    None
    """
    results = read_moonshot_csv(filepath_or_buffer)
    return from_moonshot(results, **kwargs)
