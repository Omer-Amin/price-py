For those moments when you randomly get a novel idea and want to quickly test your hypothesis.

[Documentation](https://omer-amin.github.io/pricepy/) | [PyPI](https://pypi.org/project/pricepy/)

## Installation

```bash
pip install pricepy
```

## Example

**Hypothesis:** When the daily price drops 2 standard deviations below the mean, it will rebound back to the mean within 3 days.

```python
import pricepy as ppy
import yfinance as yf

dat = yf.Ticker("MSFT")
history = dat.history(period='8mo')

candles = ppy.OHLC(history)

sma10 = ppy.sma(candles.closes, 10)
sma5 = ppy.sma(candles.closes, 5)

# Candlestick chart
ppy.candlestick(candles, overlays=[sma5, sma10])

# Daily returns distribution
daily_returns = ppy.logReturns(candles.closes)
ppy.hist(daily_returns, bins=30)
```

### Outputs:

| Candlestick + SMAs               | Daily returns histogram        |
|:--------------------------------:|:------------------------------:|
| ![Candles](./images/candle_output.png) | ![Histogram](./images/hist_output.png) |
