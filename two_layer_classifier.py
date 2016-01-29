#!/usr/bin/env python3
from neural_network import *

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


class SVM:
    def __init__(self):
        self.A = np.array([Unit(random.uniform(-3, 3)) for i in range(12)]).reshape((4, 3))
        self.d = Unit(1)
        self.nn = np.array([Unit(0), Unit(0), Unit(0), Unit(1)])
        self.circuit = Circuit()

    def forward(self, x, y):
        self.nn[3].value = 1
        # print([c.value for c in self.A.flat])
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


def main():
    svm = SVM()
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
        p = Unit(data[I][0])
        q = Unit(data[I][1])
        label = labels[I]
        svm.learn_from(p, q, label)

        if i % 25 == 0:
            print("training accuracy: " + str(evalTrainingAccuracy(data, labels, svm)))

print(10)
if __name__ == '__main__':
    main()
