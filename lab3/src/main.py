from src.network import Network


def main():
    net = Network(test_mode=True)
    print(net.predict([0.2, 0.1]))
    net.train(inputs=[0.2, 0.1], expected_outputs=[0.4, 0.6], epochs=1000)
    print(net.predict([0.2, 0.1]))


if __name__ == '__main__':
    main()
