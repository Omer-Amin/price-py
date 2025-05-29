Very simple Python library for analyzing price data

## Installation

```bash
pip install pricepy
```

## Example

The following example pulls data using `yfinance` and plots a candlestick chart overlayed with 5 and 10-day simple moving averages:

```python
import pricepy as ppy
import yfinance as yf

dat = yf.Ticker("MSFT")
history = dat.history(period='8mo')

candles = ppy.OHLC(history)

sma10 = ppy.sma(candles.closes, 10)
sma5 = ppy.sma(candles.closes, 5)

ppy.candlestick(candles, overlays=[sma5, sma10])
```

Output:

![](./images/example_output.png)

Adding the following lines will also show you the distribution of daily returns:

```python
daily_returns = ppy.logReturns(candles.closes)
ppy.hist(daily_returns, bins=30)
```

Output:

![](./images/hist1_output.png)
