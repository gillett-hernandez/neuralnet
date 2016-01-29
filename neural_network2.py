from random import random
import inspect
import abc
import math


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


class Unit:
    def __init__(self, value, grad=0):
        self.value = value
        self.grad = grad

    def __repr__(self):
        return str(self.value) + " " + str(self.grad)


class Gate(abc.ABC):
    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def backward(self):
        pass


class MulGate(Gate):
    def forward(self, x, y):
        if isinstance(x, int):
            x = Unit(x, 0)
        if isinstance(y, int):
            y = Unit(y, 0)
        self.x = x
        self.y = y
        self.out = Unit(self.x.value*self.y.value, 0)
        return self.out

    def backward(self):
        self.x.grad += self.y.value*self.out.grad
        self.y.grad += self.x.value*self.out.grad


class AddGate(Gate):
    def forward(self, x, y):
        if isinstance(x, int):
            x = Unit(x, 0)
        if isinstance(y, int):
            y = Unit(y, 0)
        self.x = x
        self.y = y
        self.out = Unit(self.x.value+self.y.value, 0)
        return self.out

    def backward(self):
        self.x.grad += 1 * self.out.grad
        self.y.grad += 1 * self.out.grad


class SigGate(Gate):
    def forward(self, x):
        # put check
        self.x = x
        self.sig = lambda x: 1 / (1 + math.exp(-x))
        self.out = Unit(self.sig(self.x.value), 0)
        return self.out

    def backward(self):
        self.x.grad += self.sig(self.x.value) * self.sig(1 - self.x.value) * self.out.grad


class PowerGate(Gate):
    def __init__(self, p):
        self.p = p

    def forward(self, x):
        self.x = x
        self.power = lambda x: pow(x, self.p)
        self.out = Unit(self.power(self.x.value), 0)
        return self.out

    def backward(self):
        self.x.grad += self.p * pow(self.out.value, self.p-1) * self.out.grad


class DivGate(Gate):
    def forward(self, x):
        self.x = x
        self.div = lambda x: 1 / abs(x)
        self.out = Unit(self.div(self.x.value), 0)
        return self.out

    def backward(self):
        self.x.grad += - sign(self.x.value) * pow(self.x.value, -2) * self.out.grad


def main():
    a = Unit(1, 0)
    b = Unit(2, 0)
    c = Unit(-3, 0)
    d = Unit(1, 0)

    # gatesetup
    mulg0 = MulGate()
    addg0 = AddGate()
    mulg1 = MulGate()
    pow2g0 = PowerGate(2)
    sigg0 = SigGate()

    def ForwardCircut():
        for v in [a, b, c, d]:
            v.grad = 0
        q = mulg0.forward(a, b)
        p = addg0.forward(q, c)
        l = mulg1.forward(p, p)
        f = pow2g0.forward(l)
        s = sigg0.forward(f)
        return s

    def backpropogation():
        # print("backprop")
        s = sigg0.out
        s.grad = 1
        sigg0.backward()
        pow2g0.backward()
        mulg1.backward()
        addg0.backward()
        mulg0.backward()

    def adjust_params(step_size=0.0001):
        a.value += step_size * a.grad
        b.value += step_size * b.grad
        c.value += step_size * c.grad
        d.value += step_size * d.grad

    for i in range(1000):
        s = ForwardCircut()
        print(s)
        backpropogation()
        adjust_params()

if __name__ == '__main__':
    main()
