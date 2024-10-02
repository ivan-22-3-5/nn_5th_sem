import numpy as np


def main():
    sm = np.array([[(i+0.25*j)**(1.5+j) for i in range(1, 4)] for j in range(1, 4)])
    print("Square matrix:", sm, sep="\n")
    print(f"Determinant: {np.linalg.det(sm)}", f"Inverse:\n{np.linalg.inv(sm)}", sep="\n")
    print('\n\n')
    v1 = np.random.randint(0, 10, 3)
    v2 = np.random.rand(3) * 10
    print(f"Vector 1: {v1}", f"Vector 2: {v2}", sep="\n")
    print(f"Multiplication: {v1.dot(v2)}", f"Sum: {v1 + v2}", sep="\n")
    print('\n\n')
    m1 = np.random.randint(0, 10, (3, 4))
    m2 = np.random.randint(0, 10, (4, 3))
    print("Matrix 1:", m1, "Matrix 2:", m2, sep="\n")
    print(f"Product of matrix 1 and matrix 2:\n{m1.dot(m2)}\n"
          f"Product of matrix 2 and matrix 1:\n{m2.dot(m1)}\n"
          f"Product of matrix 2 and vector 2:\n{m2.dot(v2)}\n"
          f"Product of vector 1 and matrix 1:\n{v1.dot(m1)}\n")


if __name__ == "__main__":
    main()
