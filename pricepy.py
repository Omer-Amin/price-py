from dataclasses import dataclass, field
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import csv
import math

@dataclass
class Candle:
    date:   str  # DD-MM-YYYY
    time:   str  # HH:MM:SS
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

class Ticker():
    def __init__(self, ticker):
        self.ticker = ticker
        self.candles = []

        filename = f"{ticker}.csv"
        with open(filename, newline='') as f:
            for row in csv.DictReader(f):
                candle = Candle(
                    date   = 'ignore for now', #row['Date'],
                    time   = 'ignore for now', #row['Time'],
                    open   = float(row['Open']),
                    high   = float(row['High']),
                    low    = float(row['Low']),
                    close  = float(row['Close']),
                    volume = float(row['Volume'])
                )
                self.candles.append(candle)

    def aslist(self, attr: str):
        values = []
        for candle in self.candles:
            values.append(getattr(candle, attr, None))
        return values

#####################
# Data Manipulation #
#####################

# Drop values not within n standard deviations from the mean
def dropsd(values: List[float], n: float = 1.5):
    mu = mean(values)
    bound = n * sd(values)
    keep = []
    for value in values:
        if value <= mu + bound and value >= mu - bound:
            keep.append(value)
    return keep

# Drop values that do not satisfy condition
def dropif(values: List[float], condition: Callback[[float, int], bool]):
    mu = mean(values)
    bound = n * sd(values)
    keep = []
    for i in range(len(values)):
        if condition(value[i], i):
            keep.append(value)
    return keep


##############
# Statistics #
##############

def pdf(values: List[float]) -> List[float]:
    x = np.array(values, dtype=float)
    z = (x - x.mean()) / x.std(ddof=0)
    pdf_vals = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    probs = pdf_vals / pdf_vals.sum()
    return probs.tolist()

# Sample standard distribution
def sd(values: List[float]) -> float:
    return statistics.stdev(values)

# Mean
def mean(values: List[float]) -> float:
    return sum(values) / len(values)

########################
# Price Trend Analysis #
########################

# Log returns
def logReturns(prices: List[float]):
    return [math.log(prices[i] / prices[i - 1]) for i in range(len(prices))]

# Arithmetic returns
def ariReturns(prices: List[float]):
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(len(prices))]

# Log return
def logReturn(old: float, new: float):
    return new / old

# Arithmetic return
def ariReturn(old: float, new: float):
    return (new - old) / old

def cumReturns(prices: List[float]):
    return [[(prices[i] - prices[0]) / prices[0] for i in range(len(prices))]]

def totReturn(prices: List[float]):
    return (prices[-1] - prices[0]) / prices[0]

#################
# Visualisation #
#################

@dataclass
class Aesthetics:
    xlabel: str = 'x'
    ylabel: str = 'y'
    title: str  = 'Plot'

def line(y: List[float], x: List[float] = [], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Line Plot'
    )   if aes == None else aes

    x = [i for i in range(len(y))] if x == [] else x
    x ,y = zip(*sorted(zip(x,y),key=lambda x: x[0]))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def scatter(y: List[float], x: List[float], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Scatter Plot'
    )  if aes == None else aes

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def histogram(values: List[float], bins: int = 10, density: bool = False, aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Historgram',
        xlabel = 'Value',
        ylabel = 'Frequency'
    )  if aes == None else aes

    plt.figure()
    plt.hist(values, bins=bins, density=density)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def distribution(values: List[float], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Distribution',
        xlabel = 'Value',
        ylabel = 'Probability'
    ) if aes == None else aes

    line(x=values, y=pdf(values), aes=aes)
