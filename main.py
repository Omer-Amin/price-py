from pricepy import *

prices = OHLC("AAPL-2015-1d").closes

hour_prices = downsample(prices, "1d", "2d")

hp = setLen(hour_prices, len(prices))

multiplot([
    ('line', prices),
    ('line', hp)
])

