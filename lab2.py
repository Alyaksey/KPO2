import numpy as np
import operator
from matplotlib import pyplot as plt


def first_sum(k):
    sum = 0.0
    for i in range(7):
        sum += np.exp(-2 * k * (i + 1))
    return sum


def second_sum(ni, k):
    sum = 0.0
    for i in range(7):
        sum += (i + 1) * ni[i] * np.exp(-k * (i + 1))
    return sum


def third_sum(k):
    sum = 0.0
    for i in range(7):
        sum += (i + 1) * np.exp(-2 * k * (i + 1))
    return sum


def fourth_sum(ni, k):
    sum = 0.0
    for i in range(7):
        sum += ni[i] * np.exp(-k * (i + 1))
    return sum


def find_K(k, ni, eps):
    while True:
        if np.abs((first_sum(k) * second_sum(ni, k) / third_sum(k)) - fourth_sum(ni, k)) < eps:
            break
        k += 0.0001
    return k


def numerator(k, ni):
    sum = 0.0
    for i in range(7):
        sum += ni[i] * (i + 1) * np.exp(-k * (i + 1))
    return sum


def denominator(k):
    sum = 0.0
    for i in range(7):
        sum += (i + 1) * np.exp(-2 * k * (i + 1))
    sum *= k
    return sum


def find_N0(k, ni):
    return numerator(k, ni) / denominator(k)


def show_first_plot(x, y, is_fourth=False):
    if is_fourth:
        plt.title('Решение для линейной аппроксимации')
    else:
        plt.title('Решение по модели Джелинского – Моранды')
    plt.xticks(range(0, 11, 1))
    plt.yticks(range(-5, 6, 1))
    plt.scatter(range(1, 8), x, color='r')
    plt.plot(range(1, 11), y)
    plt.grid(True, linestyle='-', color='0.5')
    if is_fourth:
        plt.savefig('plot2.png')
    else:
        plt.savefig('plot1.png')
    plt.show()


def LSM(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum([x[i] * y[i] for i in range(0, n)])
    sum_square_x = sum(square(x))
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_square_x - sum_x ** 2)
    b = (sum_y - a * sum_x) / n
    print('a={}'.format(a))
    print('b={}'.format(b))
    return [a * x + b for x in range(1, 11, 1)]


def show_sec_plot(x, is_square=True, is_LSM=True):
    if is_square:
        if is_LSM:
            plt.title('Значения квадратов невязок для линейной аппроксимации')
        else:
            plt.title('Значения квадратов невязок для модели Джелинского – Моранды')
    else:
        if is_LSM:
            plt.title('Значения невязок для линейной аппроксимации')
        else:
            plt.title('Значения невязок для модели Джелинского – Моранды')
    plt.scatter(range(1, 8), x, color='r')
    plt.grid(True, linestyle='-', color='0.5')
    if is_square:
        if is_LSM:
            plt.savefig('plot4_2.png')
        else:
            plt.savefig('plot3_2.png')
    else:
        if is_LSM:
            plt.savefig('plot4_1.png')
        else:
            plt.savefig('plot3_1.png')
    plt.show()


def find_residuals(x, y):
    return list(map(operator.sub, x, y))


def square(x):
    return [x ** 2 for x in x]


if __name__ == '__main__':
    k = 0.0001
    eps = 0.0001
    ni_experiment = [5, 2, 3, 1, 1, 1, 0]
    k = find_K(k, ni_experiment, eps)
    print('K={}'.format(k))
    N0 = find_N0(k, ni_experiment)
    print('N0={}'.format(N0))
    n_theor = []
    for i in range(10):
        n_theor.append(N0 * k * np.exp(-k * (i + 1)))
    show_first_plot(ni_experiment, n_theor)
    LSM = LSM(range(1, 8), ni_experiment)
    show_first_plot(ni_experiment, LSM, True)
    residuals = find_residuals(ni_experiment, n_theor)
    show_sec_plot(residuals, False, False)
    squared_residuals = square(residuals)
    show_sec_plot(squared_residuals, True, False)
    residuals_sum = sum(squared_residuals)
    print('Сумма квадратов невязок в модели Джелинского – Моранды: {}'.format(residuals_sum))
    residuals_LSM = find_residuals(ni_experiment, LSM)
    show_sec_plot(residuals_LSM, False, True)
    squared_residuals_LSM = square(residuals_LSM)
    show_sec_plot(squared_residuals_LSM, True, True)
    residuals_sum = sum(squared_residuals_LSM)
    print('Сумма квадратов невязок в линейной аппроксимации: {}'.format(residuals_sum))
