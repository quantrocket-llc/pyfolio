#
# Copyright 2019 Quantopian, Inc.
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
from __future__ import division

import warnings
from time import time

from typing import Union
import empyrical as ep
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

from . import _seaborn as sns
from . import capacity
from . import plotting
from . import pos
from . import round_trips
from . import timeseries
from . import txn
from . import utils

__all__ = [
    "create_full_tear_sheet",
    "create_capacity_tear_sheet",
    "create_interesting_times_tear_sheet",
    "create_position_tear_sheet",
    "create_returns_tear_sheet",
    "create_round_trip_tear_sheet",
    "create_simple_tear_sheet",
    "create_txn_tear_sheet",
]

def timer(msg_body, previous_time):
    current_time = time()
    run_time = current_time - previous_time
    message = "\nFinished " + msg_body + " (required {:.2f} seconds)."
    print(message.format(run_time))

    return current_time


def create_full_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame = None,
    transactions: pd.DataFrame = None,
    market_data: pd.DataFrame = None,
    benchmark_rets: 'pd.Series[float]' = None,
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
    pnl: 'pd.Series[float]' = None,
    commissions: 'pd.Series[float]' = None,
    fees: 'pd.Series[float]' = None,
    ) -> None:
    """
    Generate a number of tear sheets that are useful
    for analyzing a strategy's performance.

    - Fetches benchmarks if needed.
    - Creates tear sheets for returns, and significant events. If possible,
      also creates tear sheets for position analysis and transaction analysis.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - Time series with decimal returns.
        - Example:
          ::

            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

    positions : pd.DataFrame, optional
        Daily net position values.

        - Time series of dollar amount invested in each position and cash.
        - Days where stocks are not held can be represented by 0 or NaN.
        - Non-working capital is labelled 'cash'
        - Example:
          ::

            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375

    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.

        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
          ::

            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'

    market_data : pd.DataFrame, optional
        Daily market_data

        - DataFrame has a multi-index index, one level is dates and another is
          market_data contains volume & price, equities as columns

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

    pnl : pd.Series, optional
        Daily noncumulative pnl.

    commissions : pd.Series, optional
        Daily noncumulative commissions.

    fees : pd.Series, optional
        Daily noncumulative fees.

    Returns
    -------
    None
    """

    if (unadjusted_returns is None) and (slippage is not None) and\
       (transactions is not None):
        unadjusted_returns = returns.copy()
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    create_returns_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        live_start_date=live_start_date,
        cone_std=cone_std,
        benchmark_rets=benchmark_rets,
        bootstrap=bootstrap,
        turnover_denom=turnover_denom,
        header_rows=header_rows,
        set_context=set_context,
        pnl=pnl,
        commissions=commissions,
        fees=fees
        )

    create_interesting_times_tear_sheet(returns,
                                        benchmark_rets=benchmark_rets,
                                        set_context=set_context)

    if positions is not None:
        create_position_tear_sheet(returns, positions,
                                   hide_positions=hide_positions,
                                   set_context=set_context,
                                   sector_mappings=sector_mappings,
                                   estimate_intraday=False)

        if transactions is not None:
            create_txn_tear_sheet(returns, positions, transactions,
                                  unadjusted_returns=unadjusted_returns,
                                  estimate_intraday=False,
                                  set_context=set_context)
            if round_trips:
                create_round_trip_tear_sheet(
                    returns=returns,
                    positions=positions,
                    transactions=transactions,
                    sector_mappings=sector_mappings,
                    estimate_intraday=False)

            if market_data is not None:
                create_capacity_tear_sheet(returns, positions, transactions,
                                           market_data,
                                           liquidation_daily_vol_limit=0.2,
                                           last_n_days=125,
                                           estimate_intraday=False)

@plotting.customize
def create_simple_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame = None,
    transactions: pd.DataFrame = None,
    benchmark_rets: 'pd.Series[float]' = None,
    slippage: float = None,
    estimate_intraday: Union[bool, str] = 'infer',
    live_start_date: pd.Timestamp = None,
    turnover_denom: str = 'AGB',
    header_rows: dict[str, str] = None
    ) -> None:
    """
    Simpler version of :class:`pyfolio.create_full_tear_sheet`; generates summary performance
    statistics and important plots as a single image.

    - Plots: cumulative returns, rolling beta, rolling Sharpe, underwater,
      exposure, top 10 holdings, total holdings, long/short holdings,
      daily turnover, transaction time distribution.
    - Never accept market_data input (market_data = None)
    - Never accept sector_mappings input (sector_mappings = None)
    - Never perform bootstrap analysis (bootstrap = False)
    - Never hide posistions on top 10 holdings plot (hide_positions = False)
    - Always use default cone_std (cone_std = (1.0, 1.5, 2.0))

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - Time series with decimal returns.
        - Example:
          ::

            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902

    positions : pd.DataFrame, optional
        Daily net position values.

        - Time series of dollar amount invested in each position and cash.
        - Days where stocks are not held can be represented by 0 or NaN.
        - Non-working capital is labelled 'cash'
        - Example:
          ::

            index         'AAPL'         'MSFT'          cash
            2004-01-09    13939.3800     -14012.9930     711.5585
            2004-01-12    14492.6300     -14624.8700     27.1821
            2004-01-13    -13853.2800    13653.6400      -43.6375

    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.

        - One row per trade.
        - Trades on different names that occur at the
          same time will have identical indicies.
        - Example:
          ::

            index                  amount   price    symbol
            2004-01-09 12:18:01    483      324.12   'AAPL'
            2004-01-09 12:18:01    122      83.10    'MSFT'
            2004-01-13 14:12:23    -75      340.43   'AAPL'

    benchmark_rets : pd.Series, optional
        Daily returns of the benchmark, noncumulative.

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

    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.

        - See full explanation in txn.get_turnover.

    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.

    set_context : boolean, optional
        If True, set default plotting style context.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    if (slippage is not None) and (transactions is not None):
        returns = txn.adjust_returns_for_slippage(returns, positions,
                                                  transactions, slippage)

    always_sections = 4
    positions_sections = 4 if positions is not None else 0
    transactions_sections = 2 if transactions is not None else 0
    live_sections = 1 if live_start_date is not None else 0
    benchmark_sections = 1 if benchmark_rets is not None else 0

    vertical_sections = sum([
        always_sections,
        positions_sections,
        transactions_sections,
        live_sections,
        benchmark_sections,
    ])

    if live_start_date is not None:
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    plotting.show_perf_stats(returns,
                             benchmark_rets,
                             positions=positions,
                             transactions=transactions,
                             turnover_denom=turnover_denom,
                             live_start_date=live_start_date,
                             header_rows=header_rows)

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)

    ax_rolling_returns = plt.subplot(gs[:2, :])
    i = 2
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1

    plotting.plot_rolling_returns(returns,
                                  factor_returns=benchmark_rets,
                                  live_start_date=live_start_date,
                                  cone_std=(1.0, 1.5, 2.0),
                                  ax=ax_rolling_returns)
    ax_rolling_returns.set_title('Cumulative returns')

    if benchmark_rets is not None:
        plotting.plot_rolling_beta(returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_sharpe(returns, ax=ax_rolling_sharpe)

    plotting.plot_drawdown_underwater(returns, ax=ax_underwater)

    if positions is not None:
        # Plot simple positions tear sheet
        ax_exposures = plt.subplot(gs[i, :])
        i += 1
        ax_top_positions = plt.subplot(gs[i, :], sharex=ax_exposures)
        i += 1
        ax_holdings = plt.subplot(gs[i, :], sharex=ax_exposures)
        i += 1
        ax_long_short_holdings = plt.subplot(gs[i, :])
        i += 1

        positions_alloc = pos.get_percent_alloc(positions)

        plotting.plot_exposures(returns, positions, ax=ax_exposures)

        plotting.show_and_plot_top_positions(returns,
                                             positions_alloc,
                                             show_and_plot=0,
                                             hide_positions=False,
                                             ax=ax_top_positions)

        plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

        plotting.plot_long_short_holdings(returns, positions_alloc,
                                          ax=ax_long_short_holdings)

        if transactions is not None:
            # Plot simple transactions tear sheet
            ax_turnover = plt.subplot(gs[i, :])
            i += 1
            ax_txn_timings = plt.subplot(gs[i, :])
            i += 1

            plotting.plot_turnover(returns,
                                   transactions,
                                   positions,
                                   turnover_denom=turnover_denom,
                                   ax=ax_turnover)

            plotting.plot_txn_time_hist(transactions, ax=ax_txn_timings)

    for ax in fig.axes:
        # Matplotlib 2
        plt.setp(ax.get_xticklabels(), visible=True)

        # Matplotlib 3
        ax.tick_params(
            axis='x',
            which='major',
            bottom=True,
            top=False,
            labelbottom=True)


@plotting.customize
def create_returns_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame = None,
    transactions: pd.DataFrame = None,
    live_start_date: pd.Timestamp = None,
    cone_std: Union[float, tuple[float, float, float]] = (1.0, 1.5, 2.0),
    benchmark_rets: 'pd.Series[float]' = None,
    bootstrap: bool = False,
    turnover_denom: str = 'AGB',
    header_rows: dict[str, str] = None,
    return_fig: bool = False,
    pnl: 'pd.Series[float]' = None,
    commissions: 'pd.Series[float]' = None,
    fees: 'pd.Series[float]' = None
    ) -> Union[plt.Figure, None]:
    """
    Generate a number of plots for analyzing a strategy's returns.

    - Fetches benchmarks, then creates the plots on a single figure.
    - Plots: rolling returns (with cone), rolling beta, rolling sharpe,
      rolling Fama-French risk factors, drawdowns, underwater plot, monthly
      and annual return plots, daily similarity plots,
      and return quantile box plot.
    - Will also print the start and end dates of the strategy,
      performance statistics, drawdown periods, and the return range.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    positions : pd.DataFrame, optional
        Daily net position values.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    transactions : pd.DataFrame, optional
        Executed trade volumes and fill prices.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    live_start_date : datetime, optional
        The point in time when the strategy began live trading,
        after its backtest period.

    cone_std : float, or tuple, optional
        If float, The standard deviation to use for the cone plots.
        If tuple, Tuple of standard deviation values to use for the cone plots

        - The cone is a normal distribution with this standard deviation
          centered around a linear regression.

    benchmark_rets : pd.Series, optional
        Daily noncumulative returns of the benchmark.

        - This is in the same style as returns.

    bootstrap : boolean, optional
        Whether to perform bootstrap analysis for the performance
        metrics. Takes a few minutes longer.

    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.

        - See full explanation in txn.get_turnover.

    header_rows : dict or OrderedDict, optional
        Extra rows to display at the top of the perf stats table.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.

    pnl : pd.Series, optional
        Daily PNL of the strategy. Will only be plotted if commissions
        or fees are also provided.

    commissions : pd.Series, optional
        Daily commissions of the strategy. Will only be plotted if pnl
        is also provided.

    fees : pd.Series, optional
        Daily fees of the strategy. Will only be plotted if pnl
        is also provided.
    """

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

    plotting.show_perf_stats(returns, benchmark_rets,
                             positions=positions,
                             transactions=transactions,
                             turnover_denom=turnover_denom,
                             bootstrap=bootstrap,
                             live_start_date=live_start_date,
                             header_rows=header_rows)

    plotting.show_worst_drawdown_periods(returns)

    vertical_sections = 11

    if live_start_date is not None:
        vertical_sections += 1
        live_start_date = ep.utils.get_utc_timestamp(live_start_date)

    if benchmark_rets is not None:
        vertical_sections += 1

    if bootstrap:
        vertical_sections += 1

    if pnl is not None and commissions is not None:
        vertical_sections += 1

    if pnl is not None and fees is not None:
        vertical_sections += 1

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_rolling_returns = plt.subplot(gs[:2, :])

    i = 2
    if benchmark_rets is not None:
        ax_rolling_returns_vol_match = plt.subplot(gs[i, :],
                                                sharex=ax_rolling_returns)
        i += 1

    ax_rolling_returns_log = plt.subplot(gs[i, :],
                                         sharex=ax_rolling_returns)
    i += 1

    if pnl is not None and commissions is not None:
        ax_pnl_commissions = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1

    if pnl is not None and fees is not None:
        ax_pnl_fees = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1

    ax_returns = plt.subplot(gs[i, :],
                             sharex=ax_rolling_returns)
    i += 1
    if benchmark_rets is not None:
        ax_rolling_beta = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
        i += 1
    ax_rolling_volatility = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_rolling_sharpe = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_drawdown = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_underwater = plt.subplot(gs[i, :], sharex=ax_rolling_returns)
    i += 1
    ax_monthly_heatmap = plt.subplot(gs[i, 0])
    ax_annual_returns = plt.subplot(gs[i, 1])
    ax_monthly_dist = plt.subplot(gs[i, 2])
    i += 1
    ax_return_quantiles = plt.subplot(gs[i, :])
    i += 1

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns)
    ax_rolling_returns.set_title(
        'Cumulative returns')

    if benchmark_rets is not None:
        plotting.plot_rolling_returns(
            returns,
            factor_returns=benchmark_rets,
            live_start_date=live_start_date,
            cone_std=None,
            volatility_match=(benchmark_rets is not None),
            legend_loc=None,
            ax=ax_rolling_returns_vol_match)
        ax_rolling_returns_vol_match.set_title(
            'Cumulative returns volatility matched to benchmark')

    plotting.plot_rolling_returns(
        returns,
        factor_returns=benchmark_rets,
        logy=True,
        live_start_date=live_start_date,
        cone_std=cone_std,
        ax=ax_rolling_returns_log)
    ax_rolling_returns_log.set_title(
        'Cumulative returns on logarithmic scale')

    if pnl is not None and commissions is not None:
        plotting.plot_rolling_returns(
            pnl,
            factor_returns=commissions,
            # Don't show the cone plot on PNL graphs, it screws up the scale
            live_start_date=None,
            cone_std=cone_std,
            ax=ax_pnl_commissions,
            arithmetic=True)
        ax_pnl_commissions.set_title(
            'Cumulative PNL vs commissions')
        ax_pnl_commissions.set_ylabel('Cumulative PNL')
        y_axis_formatter = FuncFormatter(utils.pnl_format_fn)
        ax_pnl_commissions.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    if pnl is not None and fees is not None:
        plotting.plot_rolling_returns(
            pnl,
            factor_returns=fees,
            # Don't show the cone plot on PNL graphs, it screws up the scale
            live_start_date=None,
            cone_std=cone_std,
            ax=ax_pnl_fees,
            arithmetic=True)
        ax_pnl_fees.set_title(
            'Cumulative PNL vs fees')
        ax_pnl_fees.set_ylabel('Cumulative PNL')
        y_axis_formatter = FuncFormatter(utils.pnl_format_fn)
        ax_pnl_fees.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

    plotting.plot_returns(
        returns,
        live_start_date=live_start_date,
        ax=ax_returns,
    )
    ax_returns.set_title(
        'Returns')

    if benchmark_rets is not None:
        plotting.plot_rolling_beta(
            returns, benchmark_rets, ax=ax_rolling_beta)

    plotting.plot_rolling_volatility(
        returns, factor_returns=benchmark_rets, ax=ax_rolling_volatility)

    plotting.plot_rolling_sharpe(
        returns, ax=ax_rolling_sharpe)

    # Drawdowns
    plotting.plot_drawdown_periods(
        returns, top=5, ax=ax_drawdown)

    plotting.plot_drawdown_underwater(
        returns=returns, ax=ax_underwater)

    plotting.plot_monthly_returns_heatmap(returns, ax=ax_monthly_heatmap)
    plotting.plot_annual_returns(returns, ax=ax_annual_returns)
    plotting.plot_monthly_returns_dist(returns, ax=ax_monthly_dist)

    plotting.plot_return_quantiles(
        returns,
        live_start_date=live_start_date,
        ax=ax_return_quantiles)

    if bootstrap and (benchmark_rets is not None):
        ax_bootstrap = plt.subplot(gs[i, :])
        plotting.plot_perf_stats(returns, benchmark_rets,
                                 ax=ax_bootstrap)
    elif bootstrap:
        raise ValueError('bootstrap requires passing of benchmark_rets.')

    for ax in fig.axes:
        # Matplotlib 2
        plt.setp(ax.get_xticklabels(), visible=True)

        # Matplotlib 3
        ax.tick_params(
            axis='x',
            which='major',
            bottom=True,
            top=False,
            labelbottom=True)

    if return_fig:
        return fig


@plotting.customize
def create_position_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame,
    show_and_plot_top_pos: int = 2,
    hide_positions: bool = False,
    sector_mappings: Union[dict[str, str], 'pd.Series[str]'] = None,
    transactions: pd.DataFrame = None,
    estimate_intraday: Union[bool, str] = 'infer',
    return_fig: bool = False
    ) -> Union[plt.Figure, None]:
    """
    Generate a number of plots for analyzing a
    strategy's positions and holdings.

    - Plots: gross leverage, exposures, top positions, and holdings.
    - Will also print the top positions held.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    positions : pd.DataFrame
        Daily net position values.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    show_and_plot_top_pos : int, optional
        By default, this is 2, and both prints and plots the
        top 10 positions.
        If this is 0, it will only plot; if 1, it will only print.

    hide_positions : bool, optional
        If True, will not output any symbol names.
        Overrides show_and_plot_top_pos to 0 to suppress text output.

    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.

    transactions : pd.DataFrame, optional
        Prices and amounts of executed trades. One row per trade.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in :class:`pyfolio.create_full_tear_sheet`.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    if hide_positions:
        show_and_plot_top_pos = 0
    vertical_sections = 7 if sector_mappings is not None else 6

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_exposures = plt.subplot(gs[0, :])
    ax_top_positions = plt.subplot(gs[1, :], sharex=ax_exposures)
    ax_max_median_pos = plt.subplot(gs[2, :], sharex=ax_exposures)
    ax_holdings = plt.subplot(gs[3, :], sharex=ax_exposures)
    ax_long_short_holdings = plt.subplot(gs[4, :])
    ax_gross_leverage = plt.subplot(gs[5, :], sharex=ax_exposures)

    positions_alloc = pos.get_percent_alloc(positions)

    plotting.plot_exposures(returns, positions, ax=ax_exposures)

    plotting.show_and_plot_top_positions(
        returns,
        positions_alloc,
        show_and_plot=show_and_plot_top_pos,
        hide_positions=hide_positions,
        ax=ax_top_positions)

    plotting.plot_max_median_position_concentration(positions,
                                                    ax=ax_max_median_pos)

    plotting.plot_holdings(returns, positions_alloc, ax=ax_holdings)

    plotting.plot_long_short_holdings(returns, positions_alloc,
                                      ax=ax_long_short_holdings)

    plotting.plot_gross_leverage(returns, positions,
                                 ax=ax_gross_leverage)

    if sector_mappings is not None:
        sector_exposures = pos.get_sector_exposures(positions,
                                                    sector_mappings)
        if len(sector_exposures.columns) > 1:
            sector_alloc = pos.get_percent_alloc(sector_exposures)
            sector_alloc = sector_alloc.drop('cash', axis='columns')
            ax_sector_alloc = plt.subplot(gs[6, :], sharex=ax_exposures)
            plotting.plot_sector_allocations(returns, sector_alloc,
                                             ax=ax_sector_alloc)

    for ax in fig.axes:
        # Matplotlib 2
        plt.setp(ax.get_xticklabels(), visible=True)

        # Matplotlib 3
        ax.tick_params(
            axis='x',
            which='major',
            bottom=True,
            top=False,
            labelbottom=True)

    if return_fig:
        return fig


@plotting.customize
def create_txn_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    turnover_denom: str = 'AGB',
    unadjusted_returns: 'pd.Series[float]' = None,
    estimate_intraday: Union[bool, str] = 'infer',
    return_fig: bool = False
    ) -> Union[plt.Figure, None]:
    """
    Generate a number of plots for analyzing a strategy's transactions.

    Plots: turnover, daily volume, and a histogram of daily volume.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    positions : pd.DataFrame
        Daily net position values.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    turnover_denom : str, optional
        Either AGB or portfolio_value, default AGB.

        - See full explanation in txn.get_turnover.

    unadjusted_returns : pd.Series, optional
        Daily unadjusted returns of the strategy, noncumulative.
        Will plot additional swippage sweep analysis.

        - See pyfolio.plotting.plot_swippage_sleep and
          pyfolio.plotting.plot_slippage_sensitivity

    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in :class:`pyfolio.create_full_tear_sheet`.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    vertical_sections = 6 if unadjusted_returns is not None else 4

    fig = plt.figure(figsize=(14, vertical_sections * 6))
    gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.5, hspace=0.5)
    ax_turnover = plt.subplot(gs[0, :])
    ax_daily_volume = plt.subplot(gs[1, :], sharex=ax_turnover)
    ax_turnover_hist = plt.subplot(gs[2, :])
    ax_txn_timings = plt.subplot(gs[3, :])

    plotting.plot_turnover(
        returns,
        transactions,
        positions,
        turnover_denom=turnover_denom,
        ax=ax_turnover)

    plotting.plot_daily_volume(returns, transactions, ax=ax_daily_volume)

    try:
        plotting.plot_daily_turnover_hist(transactions,
                                          positions,
                                          turnover_denom=turnover_denom,
                                          ax=ax_turnover_hist)
    except ValueError:
        warnings.warn('Unable to generate turnover plot.', UserWarning)

    plotting.plot_txn_time_hist(transactions, ax=ax_txn_timings)

    if unadjusted_returns is not None:
        ax_slippage_sweep = plt.subplot(gs[4, :])
        plotting.plot_slippage_sweep(unadjusted_returns,
                                     positions,
                                     transactions,
                                     ax=ax_slippage_sweep
                                     )
        ax_slippage_sensitivity = plt.subplot(gs[5, :])
        plotting.plot_slippage_sensitivity(unadjusted_returns,
                                           positions,
                                           transactions,
                                           ax=ax_slippage_sensitivity
                                           )
    for ax in fig.axes:
        # Matplotlib 2
        plt.setp(ax.get_xticklabels(), visible=True)

        # Matplotlib 3
        ax.tick_params(
            axis='x',
            which='major',
            bottom=True,
            top=False,
            labelbottom=True)

    if return_fig:
        return fig


@plotting.customize
def create_round_trip_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    sector_mappings: Union[dict[str, str], 'pd.Series[str]'] = None,
    estimate_intraday: Union[bool, str] = 'infer',
    return_fig: bool = False
    ) -> Union[plt.Figure, None]:
    """
    Generate a number of figures and plots describing the duration,
    frequency, and profitability of trade "round trips."
    A round trip is started when a new long or short position is
    opened and is only completed when the number of shares in that
    position returns to or crosses zero.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    positions : pd.DataFrame
        Daily net position values.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    sector_mappings : dict or pd.Series, optional
        Security identifier to sector mapping.
        Security ids as keys, sectors as values.

    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in :class:`pyfolio.create_full_tear_sheet`.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    transactions_closed = round_trips.add_closing_transactions(positions,
                                                               transactions)
    # extract_round_trips requires BoD portfolio_value
    trades = round_trips.extract_round_trips(
        transactions_closed,
        portfolio_value=positions.sum(axis='columns') / (1 + returns)
    )

    if len(trades) < 5:
        warnings.warn(
            """Fewer than 5 round-trip trades made.
               Skipping round trip tearsheet.""", UserWarning)
        return

    round_trips.print_round_trip_stats(trades)

    plotting.show_profit_attribution(trades)

    if sector_mappings is not None:
        sector_trades = round_trips.apply_sector_mappings_to_round_trips(
            trades, sector_mappings)
        plotting.show_profit_attribution(sector_trades)

    fig = plt.figure(figsize=(14, 3 * 6))

    gs = gridspec.GridSpec(3, 2, wspace=0.5, hspace=0.5)

    ax_trade_lifetimes = plt.subplot(gs[0, :])
    ax_prob_profit_trade = plt.subplot(gs[1, 0])
    ax_holding_time = plt.subplot(gs[1, 1])
    ax_pnl_per_round_trip_dollars = plt.subplot(gs[2, 0])
    ax_pnl_per_round_trip_pct = plt.subplot(gs[2, 1])

    plotting.plot_round_trip_lifetimes(trades, ax=ax_trade_lifetimes)

    plotting.plot_prob_profit_trade(trades, ax=ax_prob_profit_trade)

    trade_holding_times = [x.days for x in trades['duration']]
    sns.histplot(trade_holding_times, kde=False, ax=ax_holding_time)
    ax_holding_time.set(xlabel='Holding time in days')

    sns.histplot(trades.pnl, kde=False, ax=ax_pnl_per_round_trip_dollars)
    ax_pnl_per_round_trip_dollars.set(xlabel='PnL per round-trip trade in $')

    sns.histplot(trades.returns.dropna() * 100, kde=False,
                 ax=ax_pnl_per_round_trip_pct)
    ax_pnl_per_round_trip_pct.set(
        xlabel='Round-trip returns in %')

    gs.tight_layout(fig)

    if return_fig:
        return fig


@plotting.customize
def create_interesting_times_tear_sheet(
    returns: 'pd.Series[float]',
    benchmark_rets: 'pd.Series[float]' = None,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = None,
    legend_loc: str = 'best',
    return_fig: bool = False
    ) -> Union[plt.Figure, None]:
    """
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.

    Plots: returns around the dotcom bubble burst, Lehmann Brothers' failure,
    9/11, US downgrade and EU debt crisis, Fukushima meltdown, US housing
    bubble burst, EZB IR, Great Recession (August 2007, March and September
    of 2008, Q1 & Q2 2009), flash crash, April and October 2014.

    benchmark_rets must be passed, as it is meaningless to analyze performance
    during interesting times without some benchmark to refer to.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    benchmark_rets : pd.Series
        Daily noncumulative returns of the benchmark.

        - This is in the same style as returns.

    periods: dict or OrderedDict, optional
        historical event dates that may have had significant
        impact on markets

    legend_loc : plt.legend_loc, optional
         The legend's location.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    rets_interesting = timeseries.extract_interesting_date_ranges(
        returns, periods)

    if not rets_interesting:
        warnings.warn('Passed returns do not overlap with any '
                      'interesting times.', UserWarning)
        return

    utils.print_table(pd.DataFrame(rets_interesting)
                      .describe().transpose()
                      .loc[:, ['mean', 'min', 'max']] * 100,
                      name='Stress Events',
                      float_format='{0:.2f}%'.format)

    if benchmark_rets is not None:
        returns = utils.clip_returns_to_benchmark(returns, benchmark_rets)

        bmark_interesting = timeseries.extract_interesting_date_ranges(
            benchmark_rets, periods)

    num_plots = len(rets_interesting)
    # 2 plots, 1 row; 3 plots, 2 rows; 4 plots, 2 rows; etc.
    num_rows = int((num_plots + 1) / 2.0)
    fig = plt.figure(figsize=(14, num_rows * 6.0))
    gs = gridspec.GridSpec(num_rows, 2, wspace=0.5, hspace=0.5)

    for i, (name, rets_period) in enumerate(rets_interesting.items()):
        # i=0 -> 0, i=1 -> 0, i=2 -> 1 ;; i=0 -> 0, i=1 -> 1, i=2 -> 0
        ax = plt.subplot(gs[int(i / 2.0), i % 2])

        ep.cum_returns(rets_period).plot(
            ax=ax, color='forestgreen', label='algo', alpha=0.7, lw=2)

        if benchmark_rets is not None:
            ep.cum_returns(bmark_interesting[name]).plot(
                ax=ax, color='gray', label='benchmark', alpha=0.6)
            ax.legend(['Algo',
                       'benchmark'],
                      loc=legend_loc, frameon=True, framealpha=0.5)
        else:
            ax.legend(['Algo'],
                      loc=legend_loc, frameon=True, framealpha=0.5)

        ax.set_title(name)
        ax.set_ylabel('Returns')
        ax.set_xlabel('')

    if return_fig:
        return fig


@plotting.customize
def create_capacity_tear_sheet(
    returns: 'pd.Series[float]',
    positions: pd.DataFrame,
    transactions: pd.DataFrame,
    market_data: pd.DataFrame,
    liquidation_daily_vol_limit: float = 0.2,
    trade_daily_vol_limit: float = 0.05,
    last_n_days: int = utils.APPROX_BDAYS_PER_MONTH * 6,
    days_to_liquidate_limit: int = 1,
    estimate_intraday: Union[bool, str] = 'infer',
    return_fig: bool = False
    ) -> Union[plt.Figure, None]:
    """
    Generate a report detailing portfolio size constraints set by
    least liquid tickers. Plots a "capacity sweep," a curve describing
    projected sharpe ratio given the slippage penalties that are
    applied at various capital bases.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    positions : pd.DataFrame
        Daily net position values.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    transactions : pd.DataFrame
        Prices and amounts of executed trades. One row per trade.

        - See full explanation in :class:`pyfolio.create_full_tear_sheet`.

    market_data : pd.DataFrame
        Daily market_data

        - DataFrame has a multi-index index, one level is dates and another is
          market_data contains volume & price, equities as columns

    liquidation_daily_vol_limit : float
        Max proportion of a daily bar that can be consumed in the
        process of liquidating a position in the
        "days to liquidation" analysis.

    trade_daily_vol_limit : float
        Flag daily transaction totals that exceed proportion of
        daily bar.

    last_n_days : integer
        Compute max position allocation and dollar volume for only
        the last N days of the backtest

    days_to_liquidate_limit : integer
        Display all tickers with greater max days to liquidation.

    estimate_intraday: boolean or str, optional
        Approximate returns for intraday strategies.
        See description in :class:`pyfolio.create_full_tear_sheet`.

    return_fig : boolean, optional
        If True, returns the figure that was plotted on.
    """

    positions = utils.check_intraday(estimate_intraday, returns,
                                     positions, transactions)

    print("Max days to liquidation is computed for each traded name "
          "assuming a 20% limit on daily bar consumption \n"
          "and trailing 5 day mean volume as the available bar volume.\n\n"
          "Tickers with >1 day liquidation time at a"
          " constant $1m capital base:")

    max_days_by_ticker = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market_data,
        max_bar_consumption=liquidation_daily_vol_limit,
        capital_base=1e6,
        mean_volume_window=5)
    max_days_by_ticker.index = (
        max_days_by_ticker.index.map(utils.format_asset))

    print("Whole backtest:")
    utils.print_table(
        max_days_by_ticker[max_days_by_ticker.days_to_liquidate >
                           days_to_liquidate_limit])

    max_days_by_ticker_lnd = capacity.get_max_days_to_liquidate_by_ticker(
        positions, market_data,
        max_bar_consumption=liquidation_daily_vol_limit,
        capital_base=1e6,
        mean_volume_window=5,
        last_n_days=last_n_days)
    max_days_by_ticker_lnd.index = (
        max_days_by_ticker_lnd.index.map(utils.format_asset))

    print("Last {} trading days:".format(last_n_days))
    utils.print_table(
        max_days_by_ticker_lnd[max_days_by_ticker_lnd.days_to_liquidate > 1])

    llt = capacity.get_low_liquidity_transactions(transactions, market_data)
    llt.index = llt.index.map(utils.format_asset)

    print('Tickers with daily transactions consuming >{}% of daily bar \n'
          'all backtest:'.format(trade_daily_vol_limit * 100))
    utils.print_table(
        llt[llt['max_pct_bar_consumed'] > trade_daily_vol_limit * 100])

    llt = capacity.get_low_liquidity_transactions(
        transactions, market_data, last_n_days=last_n_days)

    print("Last {} trading days:".format(last_n_days))
    utils.print_table(
        llt[llt['max_pct_bar_consumed'] > trade_daily_vol_limit * 100])

    bt_starting_capital = positions.iloc[0].sum() / (1 + returns.iloc[0])
    fig, ax_capacity_sweep = plt.subplots(figsize=(14, 6))
    plotting.plot_capacity_sweep(returns, transactions, market_data,
                                 bt_starting_capital,
                                 min_pv=100000,
                                 max_pv=300000000,
                                 step_size=1000000,
                                 ax=ax_capacity_sweep)

    if return_fig:
        return fig
