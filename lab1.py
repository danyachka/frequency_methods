import numpy as np
from utils import Plot, draw


a = -1
b = 1

t0 = 0
t1 = 1
t2 = 2

t0_c = 0
t1_c = 8
t2_c = 13
t3_c = 21
t4_c = 26

NList = [1, 3, 4, 5, 15]


# Функция квадратной волны
def f_sq(space):
    result = np.zeros(len(space))
    for i in range(len(space)):
        t = space[i]
        current = t % (t2 - t0)
        if t0 <= current < t1:
            result[i] = a
            continue
        result[i] = b

    return result


def f_c(t):
    return np.cos(t * np.pi) + np.cos(4*t*np.pi)


def f_nc(t):
    return 0.5 * np.sin(t * np.pi) + np.sin(2*t * np.pi) + 0.1


def f_r(t):
    return np.sin((t * np.pi) - 0.8) - np.cos(4*t*np.pi)


def f_complex(space):
    T = t4_c - t0_c
    result = np.zeros(len(space), dtype=complex)

    for i in range(len(space)):
        t = space[i] % T

        if t < t1_c:
            result[i] = t + 1j * (8 - t)
        elif t < t2_c:
            t -= t1_c
            result[i] = (8 - t) + 1j * (-t)
        elif t < t3_c:
            t -= t2_c
            result[i] = (3 - t) + 1j * (0.375 * t - 5)
        elif t <= t4_c:
            t -= t3_c
            result[i] = (t - 5) + 1j * (2 * t - 2)
    return result


def omega(n, T=t2-t0):
    return 2*np.pi*n/T


def calc(t, y, n):
    omg = omega(n)

    an = (2/(t2-t0)) * np.trapz(y * np.cos(omg * t), t)
    bn = (2/(t2-t0)) * np.trapz(y * np.sin(omg * t), t)

    cn = (1/(t2-t0)) * np.trapz(y * (np.cos(-omg * t) + np.sin(omg * t)), t)
    return an, bn, cn, omg


def F_G_N(t, y, label, isSquare=False):
    plots = [Plot(t, y, "t", "f(t)")]
    draw(plots, label, T=t2-t0)

    T = t2 - t0

    for N in NList:
        F = (2 / (t2 - t0)) * np.trapz(y * np.cos(omega(0) * t), t)
        G = 0

        for n in range(N + 1):
            omg = omega(n)

            an = (2 / T) * np.trapz(y * np.cos(omg * t), t)
            bn = (2 / T) * np.trapz(y * np.sin(omg * t), t)

            F += an * np.cos(omg*t) + bn * np.sin(omg*t)

        for n in range(-N - 1, N + 1):
            omg = omega(n)
            cn = (1 / T) * np.trapz(y * (np.cos(-omg * t) + 1j * np.sin(-omg * t)), t)

            G += cn * (np.cos(omg * t) + 1j * np.sin(omg * t))

        draw([(Plot(t, F, "t", "Fn(t) n = " + str(N)))], label,  T=t2-t0)
        draw([Plot(t, G, "t", "Gn(t) n = " + str(N))], label,  T=t2-t0)

    ##draw(plots, label)


def perservalEquality(f, N):
    T = 2*np.pi
    t = np.linspace(-np.pi, np.pi, 1000)

    norm_squared = np.trapz(np.abs(f(t)) ** 2, t)

    def an(n):
        return np.trapz(f(t) * np.cos(2 * np.pi * n * t / T), t) * 2 / T

    def bn(n):
        return np.trapz(f(t) * np.sin(2 * np.pi * n * t / T), t) * 2 / T

    aList = [an(i) for i in range(N + 1)]
    bList = [bn(i) for i in range(1, N + 1)]

    def cn(n):
        return np.trapz(f(t) * np.exp(-1j * 2 * np.pi * n * t / T), t) / T

    c = [cn(i) for i in range(N + 1)]

    c_sum = 2 * np.pi * np.sum(np.abs(c) ** 2)
    ab_sum = np.pi * (aList[0] ** 2 / 2 + np.sum([aList[i] ** 2 + bList[i - 1] ** 2 for i in range(1, N + 1)]))

    print(f'Norm squared: {norm_squared:.5f}, \tSum of abs(c(i))^2 = '
          f'{c_sum:.5f}, \tSum of abs(a(i))^2 + abs(b(i))^2 = {ab_sum:.5f}')


def createDefaultPlot():
    t = np.linspace(-1, 1, 1000)

    y = f_sq(t)
    label = "Квадратная функция"
    perservalEquality(f_sq, 100)
    ##F_G_N(t, y, label)

    y = f_c(t)
    label = "Чётная функция"
    ##F_G_N(t, y, label)

    y = f_nc(t)
    label = "Нечётная функция"
    F_G_N(t, y, label)

    y = f_r(t)
    label = "Случайная функция"
    ##F_G_N(t, y, label)


def createComplex():
    t = np.linspace(t0_c, t4_c, 10000)

    c = f_complex(t)
    label = "Комплексная функция"

    for N in NList:
        G = 0
        for n in range(-N - 1, N + 1):
            omg = omega(n, t4_c - t0_c)
            cn = (1 / t4_c) * np.trapz(c * np.exp(-1j * omg * t), t)
            if n == N:
                print(cn)

            G += cn * (np.exp(1j * omg * t))

        G_r = np.real(G)
        G_i = np.imag(G)
        draw([Plot(t, G_i, "t", "G_i_n(t) n = " + str(N))], label + " G_i(t)", setLimits=False)
        draw([Plot(t, G_r, "t", "G_r_n(t) n = " + str(N))], label + " G_r(t)", setLimits=False)
        draw([Plot(G_r, G_i, "t", "Gn(t) n = " + str(N))],
             label, setLimits=False)

    draw([Plot(np.real(c), np.imag(c), "", "")],
         label, setLimits=False)


if __name__ == '__main__':
    createComplex()
