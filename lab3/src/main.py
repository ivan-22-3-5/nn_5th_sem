from src.network import Network


def main():
    print(Network(2, 4, 3).predict([2, 3]))


if __name__ == '__main__':
    main()
