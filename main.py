from pricepy import *

prices = OHLC("AAPL-2015-1d").closes

ma = sma(prices, 100)

sma = setLen(ma, len(prices))

multiplot([
    ('line', prices),
    ('line', sma)
])

