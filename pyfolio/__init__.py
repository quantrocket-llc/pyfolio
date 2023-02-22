"""
Performance and risk analysis library for financial portfolios.

Functions
---------
from_zipline_csv
    Create a full tear sheet from a zipline backtest results CSV.

from_moonshot_csv
    Create a full tear sheet from a moonshot backtest results CSV.

create_full_tear_sheet
    Generate a number of tear sheets that are useful for analyzing a
    strategy's performance.

create_capacity_tear_sheet
    Generate a report detailing portfolio size constraints set by
    least liquid tickers.

create_interesting_times_tear_sheet
    Generate a number of returns plots around interesting points in time,
    like the flash crash and 9/11.

create_position_tear_sheet
    Generate a number of plots for analyzing a strategy's positions and holdings.

create_returns_tear_sheet
    Generate a number of plots for analyzing a strategy's returns.

create_round_trip_tear_sheet
    Generate a number of figures and plots describing the duration,
    frequency, and profitability of trade "round trips."

create_simple_tear_sheet
    Simpler version of `create_full_tear_sheet`; generate summary performance
    statistics and important plots as a single image.

create_txn_tear_sheet
    Generate a number of plots for analyzing a strategy's transactions.
"""
from . import utils
from . import timeseries
from . import pos
from . import txn
from . import interesting_periods
from . import capacity
from . import round_trips

from .tears import *  # noqa
from .plotting import *  # noqa
from ._version import get_versions
from .quantrocket_moonshot import * # noqa
from .quantrocket_zipline import from_zipline_csv # noqa

__version__ = get_versions()['version']
del get_versions

__all__ = [
    'utils',
    'timeseries',
    'pos',
    'txn',
    'interesting_periods',
    'capacity',
    'round_trips',
    'from_zipline_csv',
    'from_moonshot_csv'
    ]
