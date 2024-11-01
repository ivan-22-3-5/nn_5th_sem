from src.network import Network


def main():
    net = Network(test_mode=True)
    print(net._get_internal_errors([0.2, 0.1], [0.4, 0.6]))


if __name__ == '__main__':
    main()
