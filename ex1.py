import numpy as np
import matplotlib.pyplot as plt


a = -1
b = 1

t0 = 0
t1 = 1
t2 = 2


class Plot:

    def __init__(self, x, y, xLabel, yLabel):
        self.x = x
        self.y = y
        self.xLabel = xLabel
        self.yLabel = yLabel


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


def draw(plots, title):
    ax = plt.subplot(1, 1, 1)
    colors = ["r", "b", "g", "p"]
    labels = []

    T = t2 - t0

    for i in range(0, len(plots)):
        plot, color = plots[i], colors[i % len(colors)]
        x, y, xLabel, yLabel = plot.x, plot.y, plot.xLabel, plot.yLabel

        ax.plot(x, y, "-")
        ax.plot(x - T, y, "-")
        ax.plot(x + T, y, "-")

        labels.append(yLabel)

    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-1.2, 1.2)
    plt.title(title)
    plt.grid(True)
    plt.legend(labels)
    plt.show()


def omega(n):
    return 2*np.pi*n/(t2 - t0)


def calc(t, y, n):
    omg = omega(n)

    an = (2/(t2-t0)) * np.trapz(y * np.cos(omg * t), t)
    bn = (2/(t2-t0)) * np.trapz(y * np.sin(omg * t), t)

    cn = (1/(t2-t0)) * np.trapz(y * (np.cos(-omg * t) + np.sin(omg * t)), t)
    return an, bn, cn, omg


def F_G_N(t, y, label, isSquare=False):
    plots = [Plot(t, y, "t", "f(t)")]
    draw(plots, label)

    T = t2 - t0

    for N in [1, 3, 4, 5, 100]:
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

        draw([(Plot(t, F, "t", "Fn(t) n = " + str(N)))], label)
        draw([Plot(t, G, "t", "Gn(t) n = " + str(N))], label)

    ##draw(plots, label)


def perservalEquality(real, N):
    T = 2*np.pi
    t = np.linspace(-np.pi, np.pi, 1000)

    norm_squared = np.trapz(np.abs(real(t)) ** 2, t)

    def an(n):
        return np.trapz(real(t) * np.cos(2 * np.pi * n * t / T), t) * 2 / T

    def bn(n):
        return np.trapz(real(t) * np.sin(2 * np.pi * n * t / T), t) * 2 / T

    aList = [an(i) for i in range(N + 1)]
    bList = [bn(i) for i in range(1, N + 1)]

    def cn(n):
        return np.trapz(real(t) * np.exp(-1j * 2 * np.pi * n * t / T), t) / T

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
    F_G_N(t, y, label)

    y = f_c(t)
    label = "Чётная функция"
    ##F_G_N(t, y, label)

    y = f_nc(t)
    label = "Нечётная функция"
    ##F_G_N(t, y, label)

    y = f_r(t)
    label = "Случайная функция"
    ##F_G_N(t, y, label)


if __name__ == '__main__':
    createDefaultPlot()
