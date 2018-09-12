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
from .tears import create_full_tear_sheet
from quantrocket.zipline import ZiplineBacktestResult

def from_zipline_csv(filepath_or_buffer, **kwargs):
    """
    Creates a full tear sheet from a zipline backtest results CSV.

    Additional kwargs are passed to create_full_tear_sheet.

    Parameters
    ----------
    filepath_or_buffer : str or file-like object
        filepath or file-like object of the CSV

    Returns
    -------
    None
    """
    results = ZiplineBacktestResult.from_csv(filepath_or_buffer)

    # Due to 6-month rolling windows (126 days), we need at least 127 days of data
    for attr in ("returns", "benchmark_returns"):

        df = getattr(results, attr)
        if df.index.size < 127:
            num_dates = (127 - df.index.size) + 1 # +1 b/c the union index will have 1 overlapping date
            import warnings
            msg = (
                "Zipline {0} index has only {1} dates ({2} - {3}) but must "
                "have at least 127 dates for pyfolio 6-month rolling windows "
                "to chart properly, padding {0} with {4} initial zeros").format(
                    attr,
                    df.index.size,
                    df.index.min().isoformat(),
                    df.index.max().isoformat(),
                    127 - df.index.size)
            warnings.warn(msg)

            pad_idx = pd.bdate_range(end=df.index.min(), periods=num_dates)
            idx = pad_idx.union(df.index)
            setattr(results, attr, df.reindex(index=idx).fillna(0))

    return create_full_tear_sheet(
        results.returns,
        positions=results.positions,
        transactions=results.transactions,
        benchmark_rets=results.benchmark_returns,
        **kwargs)
