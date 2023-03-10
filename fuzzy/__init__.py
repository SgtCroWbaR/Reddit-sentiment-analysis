from scipy import stats
from statistics import mean
from enum import Enum, unique, auto


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FuzzyInput:
    def __init__(self, points, x_test):
        self.points = points
        self.x_test = x_test
        self.mu = self.__calcMu()

    def __calcMu(self):
        if self.x_test < self.points[0].x:
            return self.points[0].y
        elif self.x_test > self.points[-1].x:
            return self.points[-1].y

        for p1, p2 in zip(self.points, self.points[1:]):
            if p1.x <= self.x_test <= p2.x:
                delta = p2.x - p1.x
                self.x_test -= delta
                if p1.y == p2.y:
                    return p1.y
                else:
                    return self.__bell_slope(delta, self.x_test, p1.y < p2.y)

    @staticmethod
    def __bell_slope(delta, x_test, upward=True):
        if upward:
            x_test = delta - x_test
        return stats.norm.pdf(x_test, 0, delta / 3) / stats.norm.pdf(0, 0, delta / 3)


class FuzzyOutput:
    def __init__(self, points):
        self.points = points
        self.mu = 0
        self.c = self.calcC()

    def calcC(self):
        return mean([p.x for p in self.points if p.y == 1])


@unique
class LogicOp(Enum):
    AND = auto()
    OR = auto()
    XOR = auto()


class Rule:
    def __init__(self, input_list, output, operator):
        if operator == LogicOp.AND:
            output.mu = max(output.mu, min([x.mu for x in input_list]))
        else:
            output.mu = max(output.mu, max([x.mu for x in input_list]))


def defuzzyfy(outputs):
    return sum([outputs[i].mu * outputs[i].c for i in range(len(outputs))]) / sum([out.mu for out in outputs])
