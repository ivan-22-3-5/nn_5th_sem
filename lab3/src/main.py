from src.network import Network


def main():
    net = Network(2, 3, 2, 1)
    net.train(inputs=[[0.8, 0.5], [0.1, 0.3], [0.6, 0.1]], expected_outputs=[[0.45], [0.9], [0.4]], epochs=40000)
    print(net.predict([0.8, 0.5]))
    print(net.predict([0.1, 0.3]))
    print(net.predict([0.6, 0.1]))


if __name__ == '__main__':
    main()
