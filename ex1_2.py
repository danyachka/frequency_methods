import numpy as np
import matplotlib.pyplot as plt


# Функция квадратной волны
def f(t, a, b, t0, t1, t2):
    if t0 <= t < t1:
        return a
    elif t1 <= t <= t2:
        return b


# Вычисление коэффициентов Фурье
def fourier_coefficients(a, b, t0, t1, t2, N):
    T = t2 - t0
    omega0 = 2 * np.pi / T
    a0 = (a * (t1 - t0) + b * (t2 - t1)) / T

    an = np.zeros(N + 1)
    bn = np.zeros(N + 1)
    cn = np.zeros(N + 1, dtype=complex)

    for n in range(1, N + 1):
        omega_n = n * omega0
        an[n] = 2 * (a * np.sin(omega_n * t1) - a * np.sin(omega_n * t0) + b * np.sin(omega_n * t2) - b * np.sin(
            omega_n * t1)) / (T * omega_n)
        bn[n] = 2 * (a * (1 - np.cos(omega_n * t1)) - a * (1 - np.cos(omega_n * t0)) + b * (
                    1 - np.cos(omega_n * t2)) - b * (1 - np.cos(omega_n * t1))) / (T * omega_n)
        cn[n] = an[n] - 1j * bn[n]

    return a0, an, bn, cn


# Параметры функции
a = 2
b = 3
t0 = 1
t1 = 2
t2 = 3

# Количество членов ряда Фурье
N = 5

# График функции f(t)
t_values = np.linspace(0, 4, 1000)
f_values = [f(t, a, b, t0, t1, t2) for t in t_values]


# График частичной суммы Фурье
def fourier_series(t, a, b, t0, t1, t2, N):
    a0, an, bn, _ = fourier_coefficients(a, b, t0, t1, t2, N)
    series_sum = a0 / 2
    omega0 = 2 * np.pi / (t2 - t0)

    for n in range(1, N + 1):
        omega_n = n * omega0
        series_sum += an[n] * np.cos(omega_n * t) + bn[n] * np.sin(omega_n * t)

    return series_sum


series_values = [fourier_series(t, a, b, t0, t1, t2, N) for t in t_values]

# Вычисление коэффициентов Фурье
a0, an, bn, cn = fourier_coefficients(a, b, t0, t1, t2, N)

print("a0:", a0)
print("an:", an)
print("bn:", bn)
print("cn:", cn)

# Графики
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_values, f_values, label='f(t)')
plt.plot(t_values, series_values, label='Fourier Series')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('График функции f(t) и частичной суммы ряда Фурье')
plt.legend()

plt.subplot(2, 1, 2)
plt.stem(np.arange(N + 1), np.abs(cn) ** 2, label='|cn|^2')
plt.xlabel('n')
plt.ylabel('|cn|^2')
plt.title('Сумма квадратов коэффициентов Фурье')
plt.xticks(np.arange(N + 1))
plt.grid(True)

plt.tight_layout()
plt.show()
