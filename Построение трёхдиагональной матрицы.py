import numpy as np


def householder_tridiagonalization(A):
    """
    Приводит симметричную матрицу A к трехдиагональному виду T = Q^T * A * Q.
    Возвращает трехдиагональную матрицу T и матрицу преобразования Q.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # Проверка на квадратную матрицу
    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    Q = np.eye(n)
    T = A.copy()

    for k in range(n - 2):
        # Выделяем вектор x, который нужно обнулить (ниже главной поддиагонали)
        x = T[k + 1:, k]

        # Вычисляем норму вектора x
        sigma = np.linalg.norm(x)
        if sigma == 0:
            continue

        # Формируем вектор Хаусхолдера v
        v = x.copy()
        # Выбор знака для избежания потери точности
        v[0] += np.sign(x[0]) * sigma if x[0] != 0 else sigma
        v = v / np.linalg.norm(v)

        # Строим локальную матрицу отражения H_local = I - 2 * v * v^T
        H_local = np.eye(n - (k + 1)) - 2.0 * np.outer(v, v)

        # Расширяем матрицу Хаусхолдера до полного размера n x n
        H = np.eye(n)
        H[k + 1:, k + 1:] = H_local

        # Обновляем трехдиагональную матрицу T = H * T * H
        T = H @ T @ H

        # Накапливаем матрицу преобразования Q
        Q = Q @ H

    # Очищаем вычислительную погрешность (зануляем элементы вне трех диагоналей)
    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1:
                T[i, j] = 0.0

    return T, Q


# --- Пример использования ---
if __name__ == "__main__":
    # Задаем случайную симметричную матрицу
    np.random.seed(42)
    M = np.random.rand(4, 4)
    symmetric_matrix = M + M.T

    print("Исходная симметричная матрица:")
    print(np.round(symmetric_matrix, 4))

    T, Q = householder_tridiagonalization(symmetric_matrix)

    print("\nРезультат трехдиагонализации (Матрица T):")
    print(np.round(T, 4))

    # Проверка: восстанавливается ли исходная матрица как Q * T * Q^T
    reconstructed = Q @ T @ Q.T
    print("\nПроверка ортогональности (Q * T * Q^T):")
    print(np.round(reconstructed, 4))
