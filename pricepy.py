from dataclasses import dataclass, field
from typing import List, Tuple, Callable, TypeVar, Any
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import statistics

T = TypeVar('T')

##########
# Config #
##########

ALLOW_INFINITY = False
WARN_INFINITY = True

DATA_PATH = './'

#######################
# Warnings and Errors #
#######################

def _WARN_INFINITY(override: bool = False):
    if WARN_INFINITY:
        warnings.warn('An infinite value was produced'
                      '\nTo disable this warning, use \'WARN_INFINITY = False\''
        , UserWarning)
    elif override:
        warnings.warn('An infinite value was produced'
                      '\n(This is a forced warning and cannot be disabled)'
        , UserWarning)

def _ERROR_INFINITY(override: bool = False):
    if not ALLOW_INFINITY:
        raise ValueError('An infinite value was produced'
                         '\nTo allow infinities and disable this error message, use \'ALLOW_INFINITY = True\''
        )
    elif override:
        warnings.warn('An infinite value was produced'
                      '\n(This is a forced error message and cannot be disabled)'
        , UserWarning)

##################
# Data Structure #
##################

@dataclass
class Candle:
    date:   str  # DD-MM-YYYY
    time:   str  # HH:MM:SS
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

class OHLC():
    def __init__(self, ticker: str, timescale: str = None):
        self.ticker = ticker
        self.timescale = timescale
        self.candles = []
        self.dates =   []
        self.times =   []
        self.opens =   []
        self.highs =   []
        self.lows  =   []
        self.closes =  []
        self.volumes = []

        filename = f'{ticker}.csv'
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
                self.dates.append(candle.date)
                self.times.append(candle.time)
                self.opens.append(candle.open)
                self.highs.append(candle.high)
                self.lows.append(candle.low)
                self.closes.append(candle.close)
                self.volumes.append(candle.volume)

#####################
# Data Manipulation #
#####################

# Drop values not within n standard deviations from the mean
def dropsd(values: List[float], n: float = 1.5) -> List[float]:
    if n < 0:
        raise ValueError('Cannot have negative n value')

    mu = mean(values)
    bound = n * sd(values)
    keep = []
    for value in values:
        if value <= mu + bound and value >= mu - bound:
            keep.append(value)
    return keep

# Drop values that do not satisfy condition
def dropif(values: List[T], condition: Callable[[T, int], bool]) -> List[T]:
    keep = []
    for i in range(len(values)):
        if condition(value[i], i):
            keep.append(value)
    return keep

# Convert timescale of data to a lower resolution (downsamlping)
def downsample(values: List[T], old: str, new: str) -> List[T]:
    oldscale = old[-1]
    oldfactor = int(old[:-1])
    newscale = new[-1]
    newfactor = int(new[:-1])

    if oldscale == 'm' and newscale == 'h':
        step = 60 * newfactor / oldfactor
    elif oldscale == 'h' and newscale == 'd':
        step = 24 * newfactor / oldfactor
    elif oldscale == 'm' and newscale == 'd':
        step = 60 * 24 * newfactor / oldfactor
    else:
        raise ValueError(f'Cannot convert from timescale \'{old}\' to \'{new}\''
                         f'\nValid timescales are multiples of \'m\', \'h\' and \'d\''
                         f'\nYou cannot convert from a lower to higher resolution timescale')

    return [values[i] for i in range(0, len(values), int(step))]

# Set the size of a list using linear interpolation
def setLen(base: List[T], newsize: int) -> List[T]:
    if len(base) == 0:
        raise ValueError('Cannot change length of an empty list')
    if newsize < 0:
        raise ValueError('List lengths cannot be negative')

    scale = newsize / len(base)

    if scale < 1:
        # TODO: handle when len(base) > len(reference)
        raise NotImplementedError

    stretched = []
    whole = int(scale)

    for i in range(len(base) - 1):
        stretched.append(base[i])
        stretched[len(stretched):len(stretched)] = inter(base[i], base[i + 1], whole - 1)
    stretched.append(base[-1])

    j = 0
    while len(stretched) < newsize:
        index = j % len(stretched)
        stretched.insert(index, stretched[index])
        j += whole

    return stretched

###########
# Algebra #
###########

# Linear interpolation, get specified number of internal points
def inter(a: float, b: float, points: int = 1) -> List[float]:
    if points < 0:
        raise ValueError('Number of internal points cannot be less than 0')

    points += 1
    return [a + ((b - a) / points) * i for i in range(1, points)]

##############
# Statistics #
##############

def pdf(values: List[float]) -> List[float]:
    if len(values) == 0:
        raise ValueError('List must be non-empty')

    x = np.array(values, dtype=float)
    z = (x - x.mean()) / x.std(ddof=0)
    pdf_vals = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    probs = pdf_vals / pdf_vals.sum()
    return probs.tolist()

# Sample standard distribution
def sd(values: List[float]) -> float:
    if len(values) == 0:
        raise ValueError('List must be non-empty')

    return statistics.stdev(values)

# Mean
def mean(values: List[float]) -> float:
    if len(values) == 0:
        raise ValueError('List must be non-empty')

    return sum(values) / len(values)

# Correlation
def corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y):
        raise ValueError('Lists must be of the same length')
    if len(x) == 0:
        raise ValueError('Lists must be non-empty')

    return float(np.corrcoef(x, y)[0, 1])

########################
# Price Trend Analysis #
########################

# Log returns
def logReturns(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        raise ValueError('List must contain at least 2 elements')

    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            if ALLOW_INFINITY:
                returns.append(math.inf * prices[i] / abs(prices[i]))
                _WARN_INFINITY()
            else:
                _ERROR_INFINITY()
        else:
            returns.append(math.log(prices[i] / prices[i - 1]))

    return returns

# Arithmetic returns
def ariReturns(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        raise ValueError('List must contain at least 2 elements')

    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] == 0:
            if ALLOW_INFINITY:
                returns.append(math.inf * prices[i] / abs(prices[i]))
                _WARN_INFINITY()
            else:
                _ERROR_INFINITY()
        else:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])

    return returns

# Log return
def logReturn(old: float, new: float) -> float:
    if old == 0:
        if ALLOW_INFINITY:
            _WARN_INFINITY()
            return math.inf * new / abs(new)
        else:
            _ERROR_INFINITY()

    return new / old

# Arithmetic return
def ariReturn(old: float, new: float) -> float:
    if old == 0:
        if ALLOW_INFINITY:
            _WARN_INFINITY()
            return math.inf * new / abs(new)
        else:
            _ERROR_INFINITY()

    return (new - old) / old

def cumReturns(prices: List[float]) -> List[float]:
    if len(prices) < 2:
        raise ValueError('List must contain at least 2 elements')
    if prices[0] == 0:
        if ALLOW_INFINITY:
            _WARN_INFINITY()
            return [math.inf * prices[i] / abs(prices[i]) for i in range(len(prices))]
        else:
            _ERROR_INFINITY()

    return [(prices[i] - prices[0]) / prices[0] for i in range(len(prices))]

def totReturn(prices: List[float]) -> float:
    if len(prices) < 2:
        raise V('List must contain at least 2 elements')
    if prices[0] == 0:
        if ALLOW_INFINITY:
            _WARN_INFINITY()
            return math.inf * prices[-1] / abs(prices[-1])
        else:
            _ERROR_INFINITY()

    return (prices[-1] - prices[0]) / prices[0]

##########
# Groups #
##########

# Get upper, lower bounds of n evenly-widthed groups.
def grBounds(values: List[float], n: int) -> List[Tuple[float, float]]:
    if n <= 0:
        raise ValueError('Cannot have less than 1 bound')

    inorder = sorted(values)
    width = (inorder[-1] - inorder[0]) / n

    return [(inorder[0] + i * width, inorder[0] + (i + 1) * width) for i in range(n)]

# Divide values into groups according to known bounds
def grGet(values: List[float], bounds: List[Tuple[float, float]]) -> List[List[float]]:
    groups = []
    for bound in bounds:
        groups.append([])
        for value in values:
            if value <= bound[1] and value > bound[0]:
                groups[-1].append(value)
            elif len(groups) == 1 and value == bound[0]:
                groups[0].append(value)

    return groups

# Get number of values in each group
def grCount(values: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
    counts = []
    for bound in bounds:
        counts.append(0)
        for value in values:
            if (value <= bound[1] and value > bound[0]) or (len(counts) == 1 and value == bound[0]):
                counts[-1] += 1

    return counts

# Get frequency of some event in each group
def grFreq(values: List[float], bounds: List[Tuple[float, float]], event: Callable[[List[float], int], bool]) -> List[float]:
    counts = []
    for bound in bounds:
        counts.append(0)
        for i in range(len(values)):
            if (values[i] <= bound[1] and values[i] > bound[0]) or (len(counts) == 1 and values[i] == bound[0]):
                if event(values, i):
                    counts[-1] += 1

    return counts

# Get probabilites of some event in each group [Pr(event | group)]
def grProb(values: List[float], bounds: List[Tuple[float, float]], event: Callable[[List[float], int], bool]) -> List[float]:
    positive = []
    total = []
    for bound in bounds:
        positive.append(0)
        total.append(0)
        for i in range(len(values)):
            if (values[i] <= bound[1] and values[i] > bound[0]) or (len(positive) == 1 and values[i] == bound[0]):
                total[-1] += 1
                if event(values, i):
                    positive[-1] += 1

    return [positive[i] / (total[i] if total[i] > 0 else 1) for i in range(len(total))]

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
    ) if aes == None else aes

    x = [i for i in range(len(y))] if x == [] else x
    x ,y = zip(*sorted(zip(x,y),key=lambda x: x[0]))

    plt.figure()
    plt.plot(x, y)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def scat(y: List[float], x: List[float], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Scatter Plot'
    ) if aes == None else aes

    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def hist(values: List[float], bins: int = 10, density: bool = False, aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Historgram',
        xlabel = 'Value',
        ylabel ='Density' if density else 'Frequency',
    ) if aes == None else aes

    plt.figure()
    plt.hist(values, bins=bins, density=density)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def histline(values: List[float], y: List[float], density: bool = False, aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        xlabel='Value',
        ylabel='Density' if density else 'Frequency',
        title='Histogram + Line'
    ) if aes == None else aes

    bins = len(y)

    groups = GB(values, bins)

    x = [(low + high) / 2 for low, high in groups]

    fig, ax1 = plt.subplots()
    ax1.hist(values, bins=bins, density=density)
    ax1.set_xlabel(aes.xlabel)
    ax1.set_ylabel(aes.ylabel)
    ax1.tick_params(axis='y')
    ax1.set_title(aes.title)

    ax2 = ax1.twinx()
    ax2.plot(x, y, color='C1')
    ax2.set_ylabel('Line Value')
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.show()

def bar(x: List[T], heights: List[float], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        xlabel='Category',
        ylabel='Value',
        title='Bar Chart'
    ) if aes == None else aes

    plt.figure()
    plt.bar(x, heights)
    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()

def distr(values: List[float], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Distribution',
        xlabel = 'Value',
        ylabel = 'Probability'
    ) if aes == None else aes

    line(x=values, y=pdf(values), aes=aes)

def multiplot(plots: List[Tuple[str, List[float]]], x: List[float] = [], aes: Aesthetics = None) -> None:
    aes = Aesthetics(
        title = 'Multiplot'
    ) if aes == None else aes

    x = [i for i in range(len(plots[0][1]))] if x == [] else x

    plt.figure()

    try:
        for plot in plots:
            if plot[0] == 'line':
                plt.plot(x, plot[1])
            elif plot[0] == 'scatter':
                plt.scatter(x, plot[1])
            else:
                raise ValueError(f'{plot[1]} is not a valid plot for multiplot')
    except ValueError as e:
        print(e)
        raise ValueError(f'At least one of your sets of y-coordinates does not have the same length as your x-coordinates'
                         f'\nConsider using \'y = setLen(y, len(x))\' to resolve the issue')

    plt.xlabel(aes.xlabel)
    plt.ylabel(aes.ylabel)
    plt.title(aes.title)
    plt.show()
