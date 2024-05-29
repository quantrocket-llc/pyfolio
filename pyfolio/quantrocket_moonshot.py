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

from typing import Union, TextIO
import pandas as pd
import numpy as np
from quantrocket.moonshot import read_moonshot_csv
from moonchart.utils import intraday_to_daily
from .tears import create_full_tear_sheet
from .quantrocket_utils import pad_initial

__all__ = [
    "from_moonshot_csv",
]

def _get_benchmark_returns(benchmark):
    """
    Returns a Series of benchmark returns, if any. If more than one column has
    benchmark returns, uses the first.
    """
    have_benchmarks = (benchmark.fillna(0) != 0).any(axis=0)
    have_benchmarks = have_benchmarks[have_benchmarks]
    if have_benchmarks.empty:
        return None

    col = have_benchmarks.index[0]
    if len(have_benchmarks.index) > 1:
        import warnings
        warnings.warn("Multiple benchmarks found, only using first ({0})".format(col))

    benchmark_returns = benchmark[col]
    benchmark_returns.name = "benchmark"
    return benchmark_returns

def from_moonshot(
        results: pd.DataFrame,
        slippage: float = None,
        live_start_date: pd.Timestamp = None,
        sector_mappings: Union[dict[str, str], 'pd.Series[str]'] = None,
        round_trips: bool = False,
        estimate_intraday: Union[bool, str] = 'infer',
        hide_positions: bool = False,
        cone_std: Union[float, tuple[float, float, float]] = (1.0, 1.5, 2.0),
        bootstrap: bool = False,
        unadjusted_returns: 'pd.Series[float]' = None,
        turnover_denom: str = 'AGB',
        set_context: bool = True,
        header_rows: dict[str, str] = None
        ) -> None:
    """
    Creates a full tear sheet from a moonshot backtest results DataFrame.

    Parameters
    ----------
    results : DataFrame
        multiindex (Field, Date) DataFrame of backtest results

    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.

        - See txn.adjust_returns_for_slippage for more details.

    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.

    hide_positions : bool, optional
        If True, will not output any symbol names.

    round_trips: boolean, optional
        If True, causes the generation of a round trip tear sheet.

    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.

    estimate_intraday: boolean or str, optional
        Instead of using the end-of-day positions, use the point in the day
        where we have the most $ invested. This will adjust positions to
        better approximate and represent how an intraday strategy behaves.
        By default, this is 'infer', and an attempt will be made to detect
        an intraday strategy. Specifying this value will prevent detection.

    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots

        - The cone is a normal distribution with this standard deviation
          centered around a linear regression.

    bootstrap : boolean (optional)
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.

    turnover_denom : str
        Either AGB or portfolio_value, default AGB.

        - See full explanation in txn.get_turnover.

    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.

    set_context : boolean, optional
        If True, set default plotting style context.

        - See plotting.context().

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
    positions["cash"] = 1 - positions.abs().sum(axis=1)

    returns.name = "returns"
    returns = pad_initial(returns)

    fields = results.index.get_level_values("Field").unique()
    if "Benchmark" in fields:
        benchmark_rets = _get_benchmark_returns(
            results.loc["Benchmark"].astype(np.float64))
        benchmark_rets.name = "benchmark_returns"
        benchmark_rets = pad_initial(benchmark_rets)
    else:
        benchmark_rets = None

    return create_full_tear_sheet(
        returns,
        positions=positions,
        benchmark_rets=benchmark_rets,
        slippage=slippage,
        live_start_date=live_start_date,
        sector_mappings=sector_mappings,
        round_trips=round_trips,
        estimate_intraday=estimate_intraday,
        hide_positions=hide_positions,
        cone_std=cone_std,
        bootstrap=bootstrap,
        unadjusted_returns=unadjusted_returns,
        turnover_denom=turnover_denom,
        set_context=set_context,
        header_rows=header_rows
    )

def from_moonshot_csv(
        filepath_or_buffer: Union[str, TextIO],
        slippage: float = None,
        live_start_date: pd.Timestamp = None,
        sector_mappings: Union[dict[str, str], 'pd.Series[str]'] = None,
        round_trips: bool = False,
        estimate_intraday: Union[bool, str] = 'infer',
        hide_positions: bool = False,
        cone_std: Union[float, tuple[float, float, float]] = (1.0, 1.5, 2.0),
        bootstrap: bool = False,
        unadjusted_returns: 'pd.Series[float]' = None,
        turnover_denom: str = 'AGB',
        set_context: bool = True,
        header_rows: dict[str, str] = None
        ) -> None:
    """
    Create a full tear sheet from a moonshot backtest results CSV.

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        filepath or file-like object of the CSV

    slippage : int/float, optional
        Basis points of slippage to apply to returns before generating
        tearsheet stats and plots.
        If a value is provided, slippage parameter sweep
        plots will be generated from the unadjusted returns.
        Transactions and positions must also be passed.

        - See txn.adjust_returns_for_slippage for more details.

    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period. This datetime should be normalized.

    hide_positions : bool, optional
        If True, will not output any symbol names.

    round_trips: boolean, optional
        If True, causes the generation of a round trip tear sheet.

    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.

    estimate_intraday: boolean or str, optional
        Instead of using the end-of-day positions, use the point in the day
        where we have the most $ invested. This will adjust positions to
        better approximate and represent how an intraday strategy behaves.
        By default, this is 'infer', and an attempt will be made to detect
        an intraday strategy. Specifying this value will prevent detection.

    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots

        - The cone is a normal distribution with this standard deviation
          centered around a linear regression.

    bootstrap : boolean (optional)
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.

    turnover_denom : str
        Either AGB or portfolio_value, default AGB.

        - See full explanation in txn.get_turnover.

    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.

    set_context : boolean, optional
        If True, set default plotting style context.

        - See plotting.context().

    Returns
    -------
    None

    Notes
    -----
    Usage Guide:

    * Moonshot backtesting: https://qrok.it/dl/pf/moonshot-backtest
    """
    results = read_moonshot_csv(filepath_or_buffer)
    return from_moonshot(
        results,
        slippage=slippage,
        live_start_date=live_start_date,
        sector_mappings=sector_mappings,
        round_trips=round_trips,
        estimate_intraday=estimate_intraday,
        hide_positions=hide_positions,
        cone_std=cone_std,
        bootstrap=bootstrap,
        unadjusted_returns=unadjusted_returns,
        turnover_denom=turnover_denom,
        set_context=set_context,
        header_rows=header_rows)
