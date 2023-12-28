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

# To run: python3 -m unittest discover -s _tests/ -p test_quantrocket*.py -t . -v

import matplotlib as mpl
mpl.use("Agg")
import unittest
from unittest.mock import patch
import io
import pandas as pd
import pyfolio
import numpy as np

MOONSHOT_RESULTS = {
    'Field': [
        'Benchmark',
        'Benchmark',
        'Benchmark',
        'NetExposure',
        'NetExposure',
        'NetExposure',
        'Return',
        'Return',
        'Return'],
    'Date': [
        '2018-05-07',
        '2018-05-08',
        '2018-05-09',
        '2018-05-07',
        '2018-05-08',
        '2018-05-09',
        '2018-05-07',
        '2018-05-08',
        '2018-05-09'],
    'AAPL(265598)': [
        0.0,
        0.0048067,
        0.0063961,
        0.25,
        0.2,
        -0.5,
        0.0018087363324810761,
        0.0012016634262259631,
        0.0015990325181403089],
    'AMZN(3691937)': [
        0.0,
        0.0,
        0.0,
        0.25,
        0.3,
        0.5,
        0.0030345678231443185,
        -0.0012108315522391664,
        0.0022937220153353977]
}

MOONSHOT_INTRADAY_RESULTS = {
    'Field': [
        'NetExposure',
        'NetExposure',
        'NetExposure',
        'NetExposure',
        'Return',
        'Return',
        'Return',
        'Return'],
    'Date': [
        '2018-05-07',
        '2018-05-07',
        '2018-05-08',
        '2018-05-08',
        '2018-05-07',
        '2018-05-07',
        '2018-05-08',
        '2018-05-08'],
    'Time': [
        '10:00:00',
        '11:00:00',
        '10:00:00',
        '11:00:00',
        '10:00:00',
        '11:00:00',
        '10:00:00',
        '11:00:00'],
    'AAPL(265598)': [
        0.25,
        0.2,
        -0.5,
        0.4,
        0.0018087363324810761,
        0.0012016634262259631,
        0.0015990325181403089,
        -0.0015790325181403089],
    'AMZN(3691937)': [
        0.25,
        0.3,
        0.5,
        -0.25,
        0.0030345678231443185,
        -0.0012108315522391664,
        0.0022937220153353977,
        0.0062937220153353977]
}

class PyFolioFromMoonshotTestCase(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    @patch("pyfolio.quantrocket_moonshot.create_full_tear_sheet")
    def test_from_moonshot_csv(self, mock_create_full_tear_sheet):

        f = io.StringIO()
        moonshot_results = pd.DataFrame(MOONSHOT_RESULTS)
        moonshot_results.to_csv(f,index=False)
        f.seek(0)

        pyfolio.from_moonshot_csv(f)

        tear_sheet_call = mock_create_full_tear_sheet.mock_calls[0]

        _, args, kwargs = tear_sheet_call
        self.assertEqual(len(args), 1)
        returns = args[0]
        self.assertEqual(returns.index.tz.tzname(None), "UTC")
        # returns were padded to len 127 (more than 6 months=126 days)
        self.assertEqual(returns.index.size, 127)
        self.assertTrue((returns.iloc[:124] == 0).all())
        self.assertDictEqual(
            returns.iloc[124:].to_dict(),
            {
                pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'): 0.0048433041556253,
                pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'): -9.168126013200028e-06,
                pd.Timestamp('2018-05-09 00:00:00+0000', tz='UTC'): 0.0038927545334756
            })
        self.assertEqual(list(kwargs.keys()), ["positions", "benchmark_rets"])
        benchmark_rets = kwargs["benchmark_rets"]
        positions = kwargs["positions"]
        self.assertListEqual(
            positions.reset_index().to_dict(orient="records"),
            [
                {'Date': pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': 0.25,
                 'AMZN(3691937)': 0.25,
                 'cash': 0.5
                 },
                {'Date': pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': 0.2,
                 'AMZN(3691937)': 0.3,
                 'cash': 0.5
                 },
                {'Date': pd.Timestamp('2018-05-09 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': -0.5,
                 'AMZN(3691937)': 0.5,
                 'cash': 0
                 }
            ]
        )
        # benchmark_rets were also padded to len 127
        self.assertEqual(benchmark_rets.index.size, 127)
        self.assertTrue((benchmark_rets.iloc[:124] == 0).all())
        self.assertDictEqual(
            # replace nan with "nan" to allow equality comparisons
            benchmark_rets.iloc[124:].to_dict(),
            {
                pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'): 0,
                pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'): 0.0048067,
                pd.Timestamp('2018-05-09 00:00:00+0000', tz='UTC'): 0.0063961
            }
        )

    @patch("pyfolio.quantrocket_moonshot.create_full_tear_sheet")
    def test_from_moonshot_csv_no_benchmark(self, mock_create_full_tear_sheet):

        f = io.StringIO()
        moonshot_results = pd.DataFrame(MOONSHOT_RESULTS)
        moonshot_results = moonshot_results[moonshot_results.Field != "Benchmark"]
        moonshot_results.to_csv(f,index=False)
        f.seek(0)

        pyfolio.from_moonshot_csv(f)

        tear_sheet_call = mock_create_full_tear_sheet.mock_calls[0]

        _, args, kwargs = tear_sheet_call
        self.assertEqual(len(args), 1)
        returns = args[0]
        self.assertEqual(returns.index.tz.tzname(None), "UTC")
        # returns were padded to len 127 (more than 6 months=126 days)
        self.assertEqual(returns.index.size, 127)
        self.assertTrue((returns.iloc[:124] == 0).all())
        self.assertDictEqual(
            returns.iloc[124:].to_dict(),
            {
                pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'): 0.0048433041556253,
                pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'): -9.168126013200028e-06,
                pd.Timestamp('2018-05-09 00:00:00+0000', tz='UTC'): 0.0038927545334756
            })
        self.assertEqual(list(kwargs.keys()), ["positions"])
        positions = kwargs["positions"]
        self.assertListEqual(
            positions.reset_index().to_dict(orient="records"),
            [
                {'Date': pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': 0.25,
                 'AMZN(3691937)': 0.25,
                 'cash': 0.5
                 },
                {'Date': pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': 0.2,
                 'AMZN(3691937)': 0.3,
                 'cash': 0.5
                 },
                {'Date': pd.Timestamp('2018-05-09 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': -0.5,
                 'AMZN(3691937)': 0.5,
                 'cash': 0
                 }
            ]
        )

    @patch("pyfolio.quantrocket_moonshot.create_full_tear_sheet")
    def test_from_moonshot_csv_pass_kwargs(self, mock_create_full_tear_sheet):

        f = io.StringIO()
        moonshot_results = pd.DataFrame(MOONSHOT_RESULTS)
        moonshot_results.to_csv(f,index=False)
        f.seek(0)

        pyfolio.from_moonshot_csv(f, foo="bar", baz="bat")

        tear_sheet_call = mock_create_full_tear_sheet.mock_calls[0]

        _, args, kwargs = tear_sheet_call
        self.assertEqual(len(args), 1)
        returns = args[0]
        self.assertEqual(returns.index.tz.tzname(None), "UTC")
        # returns were padded to len 127 (more than 6 months=126 days)
        self.assertEqual(returns.index.size, 127)
        self.assertTrue((returns.iloc[:124] == 0).all())
        self.assertDictEqual(
            returns.iloc[124:].to_dict(),
            {
                pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'): 0.0048433041556253,
                pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'): -9.168126013200028e-06,
                pd.Timestamp('2018-05-09 00:00:00+0000', tz='UTC'): 0.0038927545334756
            })
        self.assertSetEqual(set(kwargs.keys()), {"positions", "benchmark_rets", "foo", "baz"})
        self.assertEqual(kwargs["foo"], "bar")
        self.assertEqual(kwargs["baz"], "bat")

    @patch("pyfolio.quantrocket_moonshot.create_full_tear_sheet")
    def test_from_intraday_moonshot_csv(self, mock_create_full_tear_sheet):

        f = io.StringIO()
        moonshot_results = pd.DataFrame(MOONSHOT_INTRADAY_RESULTS)
        moonshot_results.to_csv(f,index=False)
        f.seek(0)

        pyfolio.from_moonshot_csv(f)

        tear_sheet_call = mock_create_full_tear_sheet.mock_calls[0]

        _, args, kwargs = tear_sheet_call
        self.assertEqual(len(args), 1)
        returns = args[0]
        self.assertEqual(returns.index.tz.tzname(None), "UTC")
        # returns were padded to len 127 (more than 6 months=126 days)
        self.assertEqual(returns.index.size, 127)
        self.assertTrue((returns.iloc[:125] == 0).all())
        self.assertDictEqual(
            returns.iloc[125:].to_dict(),
            {
                pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'): 0.004834136029612099,
                pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'): 0.0086074440306706})
        self.assertEqual(list(kwargs.keys()), ["positions"])
        positions = kwargs["positions"]
        self.assertListEqual(
            positions.reset_index().to_dict(orient="records"),
            [
                {'Date': pd.Timestamp('2018-05-07 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': 0.25,
                 'AMZN(3691937)': 0.3,
                 'cash': 0.44999999999999996},
                {'Date': pd.Timestamp('2018-05-08 00:00:00+0000', tz='UTC'),
                 'AAPL(265598)': -0.5,
                 'AMZN(3691937)': 0.5,
                 'cash': 0}
            ]
        )
