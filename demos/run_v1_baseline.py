"""Run v1 extinction_event with base tune and print results."""
import os
os.environ["MPLBACKEND"] = "Agg"

import json
import qtradex as qx
from demos.extinction_event import ExtinctionEvent

asset, currency = "BTC", "USDT"
wallet = qx.PaperWallet({asset: 0, currency: 1})
data = qx.Data(
    exchange="kucoin",
    asset=asset,
    currency=currency,
    begin="2020-01-01",
    end="2025-01-01",
)

bot = ExtinctionEvent()
result = qx.backtest(bot, data, wallet, plot=False, show=False)
print(json.dumps(result, indent=2))
