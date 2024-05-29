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
from .tears import create_full_tear_sheet
from .quantrocket_utils import pad_initial
from quantrocket.zipline import ZiplineBacktestResult

__all__ = [
    "from_zipline_csv",
]

def from_zipline_csv(
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
        header_rows: dict[str, str] = None,
        start_date: Union[str, pd.Timestamp] = None,
        end_date: Union[str, pd.Timestamp] = None
        ) -> None:
    """
    Create a full tear sheet from a zipline backtest results CSV.

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

    start_date : str or datetime, optional
        Truncate at this start date (otherwise include entire date range)

    end_date : str or datetime, optional
        Truncate at this end date (otherwise include entire date range)

    Returns
    -------
    None

    Notes
    -----
    Usage Guide:

    * Zipline backtesting: https://qrok.it/dl/pf/zipline-backtest
    """
    results = ZiplineBacktestResult.from_csv(filepath_or_buffer)

    returns = results.returns
    returns.name = "returns"
    returns = pad_initial(returns)

    positions = results.positions
    transactions = results.transactions

    benchmark_rets = results.benchmark_returns
    if benchmark_rets is not None:
        benchmark_rets.name = "benchmark"
        benchmark_rets = pad_initial(benchmark_rets)

    if start_date:
        returns = returns.loc[start_date:]
        positions = positions.loc[start_date:]
        transactions = transactions.loc[start_date:]
        if benchmark_rets is not None:
            benchmark_rets = benchmark_rets.loc[start_date:]

    if end_date:
        returns = returns.loc[:end_date]
        positions = positions.loc[:end_date]
        transactions = transactions.loc[:end_date]
        if benchmark_rets is not None:
            benchmark_rets = benchmark_rets.loc[:end_date]

    return create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
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
