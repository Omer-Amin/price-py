from pricepy import *

prices = OHLC("AAPL").closes

hour_prices = downsample(prices, "1m", "1h")

corr(hour_prices, prices)
multiplot([('line', prices), ('line', hour_prices)])

