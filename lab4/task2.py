from collections.abc import Callable

import numpy as np
from numpy import ndarray
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction


def first_order_filter(T):
    return [1], [T, 1]

def get_first_order_w_func(T):
    return lambda v: 1 / (T*v + 1)

def special_filter(a1, a2, b1, b2):
    return np.poly1d([1, a1, a2]), np.poly1d([1, b1, b2])

def get_special_w_func(a1, a2, b1, b2):
    return lambda v: (v**2 + a1*v + a2) / (v**2 + b1*v + b2)


def fourier_transform_trapezoidal(u, dt):
    N = len(u)
    w = 2*np.pi*np.fft.fftfreq(N, dt)  # Убрал fftshift
    fourier = np.fft.fft(u)
    return w, fourier


def plot_signals(
        *, t: ndarray,
        g: ndarray,
        u: ndarray,
        filtered_u: ndarray,
        filter_label: str,
        transfer_func: TransferFunction,
        w_func: Callable
):
    w, g_fourier = fourier_transform_trapezoidal(g, t[1] - t[0])
    _, u_fourier = fourier_transform_trapezoidal(u, t[1] - t[0])
    _, filtered_u_fourier = fourier_transform_trapezoidal(filtered_u, t[1] - t[0])

    filtered_by_w_u_fourier = w_func(1j*w) * u_fourier
    filtered_by_w_u = np.fft.ifft(filtered_by_w_u_fourier).real

    plt.figure(figsize=(12, 7))
    size = (2, 2)
    g_alpa = 0.5
    plot_alpha = 0.7

    # сигналы
    plt.subplot(size[0], size[1], 1)
    plt.plot(t, u, label=f'u(t)',
             color='green', alpha=plot_alpha)
    plt.plot(t, filtered_u, label='u_ф(t)',
             color='tomato', alpha=plot_alpha, linewidth=2.5)
    plt.plot(t, g, label=f'g(t)', alpha=g_alpa, color='blue')
    plt.title(filter_label)
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()

    # сигналы
    plt.subplot(size[0], size[1], 2)
    plt.plot(t, filtered_u, label='u_ф(t)',
             color='tomato', linewidth=3.5)
    plt.plot(t, filtered_by_w_u, label='F^-1{W(wi)*u^(w)}',
             color='green')
    plt.title("Сравнение u_ф(t) и F^-1{W(wi)*u^(w)}")
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()

    # АЧХ фильтра
    plt.subplot(size[0], size[1], 3)
    w_freq_h, h = signal.freqresp(transfer_func, n = 25000)
    plt.plot(w_freq_h, abs(h), label='АЧХ фильтра')
    plt.axhline(2**-0.5, t[0], t[-1], color='tomato', linestyle='--')
    plt.title('АЧХ фильтра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Модули Фурье-образов
    n = len(w)
    w = w[:n//2]
    plt.subplot(size[0], size[1], 4)
    plt.semilogy(w, np.abs(g_fourier[:n//2]), label='|g^(w)|',
                 alpha=1, color='blue')
    plt.semilogy(w, np.abs(u_fourier[:n//2]), label='|u^(w)|',
                 color='green', alpha=plot_alpha)
    plt.semilogy(w, np.abs(filtered_u_fourier[:n//2]), label='|F{u_ф(w)}|',
                 color='y', alpha=plot_alpha)
    plt.semilogy(w, np.abs(filtered_by_w_u_fourier[:n//2]), label='|W(wi)*u^(w)|',
                 color='tomato', alpha=plot_alpha)
    plt.title('Модули Фурье-образов')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()

    plt.tight_layout()
    plt.show()

def g_func(a, t1, t2, t):
    g = np.where((t >= t1) & (t <= t2), a, 0)
    return g


def main():
    # Параметры сигнала
    a = 2
    b = 1
    c = 5
    d = 100

    t = np.linspace(0, 12, 1000)

    # g
    t1, t2 = 3, 7
    g = g_func(a, t1, t2, t)

    def default():
        u = g + b * (np.random.rand(len(t)) - 0.5)

        for T in [
            0.1
            # 0.01, 0.1, 0.5, 2
                  ]:
            num, den = first_order_filter(T)
            transfer_func = signal.TransferFunction(num, den)
            _, filtered, _ = signal.lsim(transfer_func, U=u, T=t)
            plot_signals(
                g=g, t=t, u=u, filtered_u=filtered,
                filter_label=f'Фильтр первого порядка({T=})',
                transfer_func=transfer_func,
                w_func=get_first_order_w_func(T),
            )

    def special():
        w_0 = 100
        a2, b2 = w_0**2, w_0**2
        a1 = 0

        u = g + c * np.sin(d * t)
        for b1 in [10, 30, 100, 150]:
            # Применяем специальный фильтр
            num, den = special_filter(a1, a2, b1, b2)
            transfer_func = signal.TransferFunction(num, den)
            _, filtered, _ = signal.lsim(transfer_func, U=u, T=t)

            # Строим графики для специального фильтра
            plot_signals(
                g=g, t=t, u=u, filtered_u=filtered,
                filter_label=f'Режекторный полосовой фильтр ({a1=}, {a2=}, {b1=}, {b2=})',
                transfer_func=transfer_func,
                w_func=get_special_w_func(a1, a2, b1, b2),
            )

    # default()
    special()


if __name__ == '__main__':
    main()
