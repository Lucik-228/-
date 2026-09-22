import numpy as np


def lu_decomposition(A):
    """Выполняет LU-разложение матрицы A (A = L * U)."""
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            # Вычисление элементов матрицы U
            s = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - s

        for k in range(i + 1, n):
            # Вычисление элементов матрицы L
            s = sum(L[k][j] * U[j][i] for j in range(i))
            L[k][i] = (A[k][i] - s) / U[i][i]

    return L, U


def lu_solve(L, U, b):
    """Решает систему AX = b, используя готовые матрицы L и U."""
    n = L.shape[0]

    # 1. Прямой ход: решаем LY = b
    y = np.zeros(n)
    for i in range(n):
        s = sum(L[i][j] * y[j] for j in range(i))
        y[i] = b[i] - s

    # 2. Обратный ход: решаем UX = x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (y[i] - s) / U[i][i]

    return x


# --- Пример использования ---
A = np.array([[2, 1, 4],
              [1, 5, -3],
              [3, -2, 2]], dtype=float)

b1 = np.array([16, -1, 6], dtype=float)
b2 = np.array([6, 12, 1], dtype=float)  # Вторая правая часть

# 1. Раскладываем матрицу ОДИН раз
L, U = lu_decomposition(A)

# 2. Быстро решаем для разных векторов b
x1 = lu_solve(L, U, b1)
x2 = lu_solve(L, U, b2)

print("Решение для b1:", x1)
print("Решение для b2:", x2)
