import math


class SignalBase:
    def __repr__(self):
        r = f", reason={self.reason}" if getattr(self, "reason", None) else ""
        return f"{type(self).__name__}(profit={self.profit}, price={self.price}, unix={self.unix}{r})"

class Buy(SignalBase):
    def __init__(self, price=None, maxvolume=math.inf, reason=None):
        self.maxvolume = maxvolume
        self.price = price
        self.unix = 0
        self.profit = 0
        self.is_override = True
        self.reason = reason


class Sell(SignalBase):
    def __init__(self, price=None, maxvolume=math.inf, reason=None):
        self.maxvolume = maxvolume
        self.price = price
        self.unix = 0
        self.profit = 0
        self.is_override = True
        self.reason = reason


class Thresholds(SignalBase):
    def __init__(self, buying, selling, maxvolume=math.inf):
        self.maxvolume = maxvolume
        self.price = None
        self.unix = 0
        self.profit = 0
        self.buying = buying
        self.selling = selling


class Hold(SignalBase):
    """
    AKA Cancel All Orders
    """
