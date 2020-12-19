import numpy as np
import mynet


def main():
    X = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]
    y = [
        [0, 0],
        [1, 0],
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 1]
    ]
    net = mynet.NeuralNetwork(np.array(X), np.array(y),lr=0.01)
    net.train(30000)
    for q in X:
        print(net.think(q), "should be", y[X.index(q)])


if __name__ == '__main__':
    main()
