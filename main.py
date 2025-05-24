from pricepy import *

thing = Ticker("AAPL")

lr = logReturns(thing.aslist("close"))

line(
    x=lr,
    y=pdf(lr)
)
