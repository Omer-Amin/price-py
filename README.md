For those moments when you randomly get a novel idea and want to test your hypothesis.

[Documentation](https://omer-amin.github.io/pricepy/) | [PyPI](https://pypi.org/project/pricepy/)

## Installation

```bash
pip install pricepy
```

## Example

The following example analyses whether moving average crossovers are correlated with average hourly returns.

```python
import pricepy as pr
import yfinance as yf

# Fetch data with yfinance
dat     = yf.Ticker("AAPL")
history = dat.history(period="30d", interval="1h")

# Extract candle data
candles = pr.OHLC(history)
closes  = candles.closes
returns = pr.logReturns(candles.closes)

# Compute moving averages
sma10 = pr.sma(closes, 10)
sma5  = pr.sma(closes, 5)

# Split candlestick data by condition
condition = pr.Condition(sma10, '<', sma5)
pos, neg  = pr.split(returns, condition)

# Compute distributions
dPos = pr.Distribution(pos)
dNeg = pr.Distribution(neg)

# Equal means hypothesis testing
pr.wttest(dPos.x, dNeg.x)

# Plot distributions
pr.multiplot([
    ('sma10 < sma5', 'mountain', dPos.x, dPos.y),
    ('sma10 > sma5', 'mountain', dNeg.x, dNeg.y)
], disp=pr.Display(xlabel="Return", ylabel="Probability", title="Distributions"))

# Plot candles
pr.candlestick(candles, overlays=[sma5, sma10], labels=['sma5', 'sma10'])
```

### Output:

| Candlestick + SMAs               | Distributions                  |
|:--------------------------------:|:------------------------------:|
| ![Candles](./images/candle_output.png) | ![Histogram](./images/dist_output.png) |

```bash
Welch's t-test: 0.000 < 0.05 => Reject null hypothesis, means are not statistically equal
```

