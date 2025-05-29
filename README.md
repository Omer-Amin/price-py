Very simple Python library for analyzing price data

## Installation

```bash
pip install pricepy
```

## Example

Example with `yfinance`:

```python
import pricepy as ppy
import yfinance as yf

dat = yf.Ticker("MSFT")
history = dat.history(period='10mo')

candles = ppy.OHLC(history)

sma10 = ppy.sma(candles.closes, 10)
sma5 = ppy.sma(candles.closes, 5)

ppy.candlestick(candles, overlays=[sma5, sma10])
```
