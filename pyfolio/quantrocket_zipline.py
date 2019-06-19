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
from .quantrocket_utils import pad_initial
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

    returns = results.returns
    returns.name = "returns"
    returns = pad_initial(returns)

    benchmark_rets = results.benchmark_returns
    if benchmark_rets is not None:
        benchmark_rets.name = "benchmark"
        benchmark_rets = pad_initial(benchmark_rets)

    return create_full_tear_sheet(
        returns,
        positions=results.positions,
        transactions=results.transactions,
        benchmark_rets=benchmark_rets,
        **kwargs)
