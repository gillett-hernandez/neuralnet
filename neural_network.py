#!/usr/bin/env python3
import random
import math
import numpy as np
from itertools import chain


class Unit:
    def __init__(self, value, grad=0):
        self.value = value
        self.grad = grad

    def __add__(self, other):
        return Unit(self.value + (other.value if isinstance(other, Unit) else other))

    def __radd__(self, other):
        return Unit(self.value + (other.value if isinstance(other, Unit) else other))

    def __mul__(self, other):
        return Unit(self.value * (other.value if isinstance(other, Unit) else other))

    def __repr__(self):
        return str(self.value) + " " + str(self.grad)

    def update(self, d=0.0001):
        self.value += self.grad * d


class Gate:
    def __init__(self):
        self.out = None

    def forward(self):
        pass

    def backward(self):
        pass


class MulGate(Gate):
    """multiply gate"""
    def forward(self, x, y):
        if isinstance(x, int):
            x = Unit(x, 0)
        if isinstance(y, int):
            y = Unit(y, 0)
        self.x = x
        self.y = y
        self.out = Unit(x.value*y.value, 0)
        return self.out

    def backward(self):
        self.x.grad += self.y.value*self.out.grad
        self.y.grad += self.x.value*self.out.grad


class AddGate(Gate):
    """add gate"""
    def forward(self, x, y):
        if isinstance(x, int):
            x = Unit(x, 0)
        if isinstance(y, int):
            y = Unit(y, 0)
        self.x = x
        self.y = y
        self.out = Unit(x.value+y.value, 0)
        return self.out

    def backward(self):
        self.x.grad += 1 * self.out.grad
        self.y.grad += 1 * self.out.grad


class SigGate(Gate):
    """sigmoid gate"""
    def forward(self, x):
        # put check
        self.sig = lambda x: 1 / (1 + math.exp(-x))
        self.x = x
        self.out = Unit(self.sig(x.value), 0)
        return self.out

    def backward(self):
        self.x.grad += self.out.value * (1 - self.out.value) * self.out.grad


class PowerGate(Gate):
    """power gate"""
    def __init__(self, p):
        self.p = p

    def forward(self, x):
        self.power = lambda x: pow(x, self.p)
        self.x = x
        self.out = Unit(self.power(x.value), 0)
        return self.out

    def backward(self):
        self.x.grad += self.p * pow(self.out.value, self.p-1) * self.out.grad


class DivGate(Gate):
    """div gate"""
    def forward(self, x):
        self.div = lambda x: 1 / x
        self.x = x
        self.out = Unit(self.div(x.value), 0)
        return self.out

    def backward(self):
        self.x.grad += - pow(self.x.value, -2) * self.out.grad


class OverflowGate(Gate):
    """overflow gate. activates with a 1 if the sum of inputs is bigger than a certain threshold"""
    def __init__(self, l):
        self.l = l

    def forward(self, *args):
        self.args = args
        self.over = lambda args: int(sum(args).value > self.l)
        self.out = Unit(self.over(args))
        return self.out

    def compute(self, in_, o, d_o):
        return d_o

    def backward(self):
        d_o = self.out.grad
        o = self.out.value
        if self.over(self.args):
            for input in self.args:
                input.grad = self.compute(input, o, d_o)


class DotGate(Gate):
    def forward(self, X, Y):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y
        self.out = Unit(0)
        for x, y in zip(X, Y):
            assert isinstance(x, Unit) and isinstance(y, Unit), (X, Y)
            self.out += x*y
        return self.out

    def backward(self):
        d = self.out.grad
        for x, y in zip(self.X, self.Y):
            x.grad += y.value * d
            y.grad += x.value * d


class MaxGate(Gate):
    def forward(self, x):
        self.x = x
        self.out = Unit(max(0, self.x.value))
        return self.out

    def backward(self):
        d = self.out.grad
        self.x.grad = (0 if self.x.value == 0 else d)


class Circuit:
    def __init__(self):
        self.g0 = DotGate()
        self.g1 = MaxGate()
        self.g2 = DotGate()
        self.g3 = MaxGate()
        self.g4 = DotGate()
        self.g5 = MaxGate()
        self.g6 = DotGate()

    def forward(self, X, A, N, d):
        o0 = self.g0.forward(A[0], X[0])
        N[0] = self.g1.forward(o0)
        o1 = self.g2.forward(A[1], X[1])
        N[1] = self.g3.forward(o1)
        o2 = self.g4.forward(A[2], X[2])
        N[2] = self.g5.forward(o2)
        assert N[3].value == 1, N
        self.out = self.g6.forward(N, np.array(list(A[3])+d))
        return self.out

    def backward(self):
        # print("backprop")
        self.g6.backward()
        self.g5.backward()
        self.g4.backward()
        self.g3.backward()
        self.g2.backward()
        self.g1.backward()
        self.g0.backward()


class two_layer_SVM:
    def __init__(self):
        self.A = np.array([Unit(random.uniform(-3, 3)) for i in range(12)]).reshape((4, 3))
        self.d = Unit(1)
        self.nn = np.array([Unit(0), Unit(0), Unit(0), Unit(1)])
        self.circuit = Circuit()

    def forward(self, x, y):
        self.nn[3].value = 1
        print([c.value for c in self.A.flat])
        X = np.tile([x, y, Unit(1)], (3, 1))
        self.out = self.circuit.forward(X, self.A, self.nn, [self.d])
        return self.out

    def backward(self, label):
        self.nn[3].value = 1
        for v in chain(self.A.flat, self.nn, [self.d]):
            v.grad = 0

        if (label == 1 and self.out.value < 1):
            pull = 1
        elif (label == -1 and self.out.value > -1):
            pull = -1
        else:
            pull = 0

        self.circuit.out.grad = pull
        self.circuit.backward()
        for el in chain(self.A[:4, :2].flat, [self.A[3, 2]]):
            el.grad += -el.value
        #  print(self.A[0, 0].grad, self.d.grad)

    def learn_from(self, x, y, label):
        self.forward(x, y)
        self.backward(label)
        self.adjust_params()

    def adjust_params(self, step_size=0.01):
        for v in chain(self.A.flat, self.nn, [self.d]):
            # print("adjust_params " + str(type(v)))
            v.update(step_size)
        self.nn[3].value = 1


def evalTrainingAccuracy(data, labels, svm):
    num_correct = 0
    for i, datapoint in enumerate(data):
        x = Unit(datapoint[0])
        y = Unit(datapoint[1])
        true_label = labels[i]
        predicted_label = 1 if svm.forward(x, y).value > 0 else -1
        if(predicted_label == true_label):
            num_correct += 1

    return num_correct / len(data)


def two_layer_classifier():
    svm = two_layer_SVM()
    data = [[+1.2, +0.7],
            [-0.3, -0.5],
            [+3.0, +0.1],
            [-0.1, -1.0],
            [-1.0, +1.1],
            [+2.1, -3.0]]
    labels = [+1,
              -1,
              +1,
              -1,
              -1,
              +1]
    for i in range(1000):
        I = random.randrange(0, len(data))
        x = Unit(data[I][0])
        y = Unit(data[I][1])
        label = labels[I]
        svm.learn_from(x, y, label)

        if i % 25 == 0:
            print("training accuracy: " + str(evalTrainingAccuracy(data, labels, svm)))
