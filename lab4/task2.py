import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def first_order_filter(T):
    return [1], [T, 1]


def special_filter(T1, T2, T3):
    return np.poly1d([T1**2, 2*T1, 1]), np.poly1d([T2*T3, T2 + T3, 1])


def fourier_transform_trapezoidal(u, dt, t):
    N = len(u)
    v = 2 * np.pi * np.fft.fftfreq(N, dt)
    v = np.fft.fftshift(v)
    fourier = np.fft.fft(u)

    return v, np.abs(fourier)


def plot_signals(t, u, filtered_u, filter_type, T_values=None, a=-1):
    plt.figure(figsize=(12, 8))

    size = (2, 2)

    # Исходный сигнал
    plt.subplot(size[0], size[1], 1)
    plt.plot(t, u, label=f'Исходный сигнал')
    plt.title(f'Исходный сигнал{f" (a = {a})" if a != -1 else ""}')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Фильтрованный сигнал
    plt.subplot(size[0], size[1], 2)
    plt.plot(t, filtered_u, label='Фильтрованный сигнал', color='orange')
    if filter_type == 'Фильтр первого порядка':
        tag = f'Фильтрованный сигнал ({filter_type} (T = {T_values[0]}))'
    else:
        tag = 'Фильтрованный сигнал ({})'.format(filter_type)
    plt.title(tag)
    plt.xlim((0, 10))
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Модули Фурье-образов
    plt.subplot(size[0], size[1], 3)
    #v, u_fourier = signal.periodogram(u)
    v, u_fourier = fourier_transform_trapezoidal(u, t[1] - t[0], t)
    plt.semilogy(v, u_fourier, label='Исходный сигнал')
    v, filtered_u_fourier = fourier_transform_trapezoidal(filtered_u, t[1] - t[0], t)
    #v, filtered_u_fourier = signal.periodogram(filtered_u)
    plt.semilogy(v, filtered_u_fourier, label='Фильтрованный сигнал', color='orange')
    plt.title('Модули Фурье-образов')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()

    # АЧХ фильтра
    plt.subplot(size[0], size[1], 4)

    if filter_type == 'Фильтр первого порядка':
        T = T_values[0]
        num, den = first_order_filter(T)
        filter_ = signal.TransferFunction(num, den)
    else:
        #T1, T2, T3 = filter_type.split('(')[1].split(')')[0].split(',')
        T1, T2, T3 = T_values
        num, den = special_filter(T1, T2, T3)
        filter_ = signal.TransferFunction(num, den)

    v, h = signal . freqresp(filter_)
    plt.plot(v, abs(h), label='АЧХ фильтра')
    plt.title('АЧХ фильтра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Влияние постоянной времени T
    # if filter_type == 'Фильтр первого порядка':
    #     plt.subplot(size[0], size[1], 5)
    #     for T_val, filtered_sig in T_values.items():
    #         plt.plot(t, filtered_sig, label='T={}'.format(T_val))
    #     plt.title('Фильтрованный сигнал (Разные T)')
    #     plt.xlabel('Время')
    #     plt.ylabel('Амплитуда')
    #     plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Параметры сигнала
    a = 10
    b = 0.5
    c = 0.5
    d = 10

    t = np.linspace(0, 10, 500)

    # g
    t1, t2 = 3, 7
    g = np.zeros_like(t)
    for i, thisT in enumerate(t):
        if t1 <= thisT <= t2:
            g[i] = a

    def default():
        u1 = g + b * (np.random.rand(len(t)) - 0.5)

        T_0 = 0.2
        T_1 = 0.5
        T_2 = 1.5

        num, den = first_order_filter(T_0)
        filter_ = signal.TransferFunction(num, den)
        _, y1, _ = signal.lsim(filter_, U=u1, T=t)

        num1, den1 = first_order_filter(T_1)
        filter_ = signal.TransferFunction(num1, den1)
        _, y2, _ = signal.lsim(filter_, U=u1, T=t)

        num2, den2 = first_order_filter(T_2)
        filter_ = signal.TransferFunction(num2, den2)
        _, y3, _ = signal.lsim(filter_, U=u1, T=t)

        T_values = {T_0: y1,
                    T_1: y2,
                    T_2: y3}

        for T in T_values.keys():
            plot_signals(t, u1, T_values[T], 'Фильтр первого порядка', T_values=[T], a=a)

    def special():
        u2 = g + c * np.sin(d * t)
        T1 = 0.13
        T2 = 0.4
        T3 = 0.5

        # Применяем специальный фильтр
        num, den = special_filter(T1, T2, T3)
        filter_ = signal.TransferFunction(num, den)
        _, filtred, _ = signal.lsim(filter_, U=u2, T=t)

        # Строим графики для специального фильтра
        plot_signals(t, u2, filtred, 'T1={}, T2={}, T3={}'.format(T1, T2, T3),
                     T_values=[T1, T2, T3])

    #default()
    special()


if __name__ == '__main__':
    main()
