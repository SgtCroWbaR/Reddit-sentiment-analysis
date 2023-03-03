from scipy import stats
from statistics import mean
from enum import Enum, unique, auto
from collections import namedtuple
Point = namedtuple("Point", ['x', 'y'])


class FuzzyInput:
    def __init__(self, name, points, x_test):
        self.name = name
        self.points = points
        self.x_test = x_test
        self.mu = self.__calcMu()

    def __calcMu(self):
        if self.x_test <= self.points[0].x:
            return self.points[0].y
        elif self.x_test >= self.points[-1].x:
            return self.points[-1].y

        for p1, p2 in zip(self.points, self.points[1:]):
            if p1.x <= self.x_test <= p2.x:
                return FuzzyInput.line_through_two_points(p1, p2)(self.x_test)

    # TODO: prepraviti da racuna belovu krivu
    #  stats.norm.pdf()

    @staticmethod
    def line_through_two_points(a, b):
        c = (b.x - a.x) / (b.y - a.y)
        return lambda x_test: (x_test - a.x) / c + a.y


class FuzzyOutput:
    def __init__(self, name, points):
        self.name = name
        self.points = points
        self.mu = 0
        self.c = self.calcC()

    #TODO: provetiti da li je ovo tacno
    def calcC(self):
        return mean([p.x for p in self.points if p.y == 1])


@unique
class LogicOp(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()


class Rule:
    def __init__(self, input1, input2, output, operator):
        if operator == LogicOp.AND:
            output.mu = max(output.mu, min(input1.mu, input2.mu))
        else:
            output.mu = max(output.mu, max(input1.mu, input2.mu))
