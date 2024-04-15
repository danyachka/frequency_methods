import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def first_order_filter(T):
    return np.poly1d([1]), np.poly1d([1, T])


def special_filter(T1, T2, T3):
    #return np.poly1d([1, 2*T1, T1**2]), np.poly1d([1, T2 + T3, T2*T3]) #np.polymul([T2, 1], [T3, 1])
    return [1, 2*T1, T1**2], [1, T2 + T3, T2*T3] # np.polymul([T2, 1], [T3, 1])


def plot_signals(t, u, filtered_u, filter_type, T_values=None):
    plt.figure(figsize=(12, 10))

    if filter_type == 'Фильтр первого порядка':
        size = (3, 2)
    else:
        size = (2, 2)

    # Исходный сигнал
    plt.subplot(size[0], size[1], 1)
    plt.plot(t, u, label='Исходный сигнал')
    plt.title('Исходный сигнал')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Фильтрованный сигнал
    plt.subplot(size[0], size[1], 2)
    plt.plot(t, filtered_u, label='Фильтрованный сигнал', color='orange')
    plt.title('Фильтрованный сигнал ({})'.format(filter_type))
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Модули Фурье-образов
    plt.subplot(size[0], size[1], 3)
    v, u_fourier = signal.periodogram(u)
    plt.semilogy(v, u_fourier, label='Исходный сигнал')
    v, filtered_u_fourier = signal.periodogram(filtered_u)
    plt.semilogy(v, filtered_u_fourier, label='Фильтрованный сигнал', color='orange')
    plt.title('Модули Фурье-образов')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()

    # АЧХ фильтра
    plt.subplot(size[0], size[1], 4)

    if filter_type == 'Фильтр первого порядка':
        T = list(T_values.keys())[0]
        num, den = first_order_filter(T)
    elif 'Специальный фильтр' in filter_type:
        #T1, T2, T3 = filter_type.split('(')[1].split(')')[0].split(',')
        T1, T2, T3 = T_values
        num, den = special_filter(T1, T2, T3)

    v, h = signal.freqz(num, den)
    plt.plot(v, abs(h), label='АЧХ фильтра')
    plt.title('АЧХ фильтра')
    plt.xlabel('Частота')
    plt.ylabel('Амплитуда')
    plt.legend()

    # Влияние постоянной времени T
    if filter_type == 'Фильтр первого порядка':
        plt.subplot(size[0], size[1], 5)
        for T_val, filtered_sig in T_values.items():
            plt.plot(t, filtered_sig, label='T={}'.format(T_val))
        plt.title('Фильтрованный сигнал (Разные T)')
        plt.xlabel('Время')
        plt.ylabel('Амплитуда')
        plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Параметры сигнала
    a = 1.3
    b = 0.5  # Амплитуда случайной составляющей
    c = 0.5  # Амплитуда синусоидальной составляющей
    d = 1  # Частота синусоидальной составляющей

    # Время
    t = np.linspace(0, 10, 1000)

    # g
    t1, t2 = 3, 7
    g = np.zeros_like(t)
    for i, thisT in enumerate(t):
        if t1 <= thisT <= t2:
            g[i] = a

    def default():
        # Сигнал u
        u1 = g + b * (np.random.rand(len(t)) - 0.5)

        T = -0.9
        T_1 = -0.5
        T_2 = -0.7
        num, den = first_order_filter(T)
        num1, den1 = first_order_filter(T_1)
        num2, den2 = first_order_filter(T_2)
        T_values = {T: signal.filtfilt(num, den, u1),
                    T_1: signal.filtfilt(num1, den1, u1),
                    T_2: signal.filtfilt(num2, den2, u1)}

        # Строим графики для фильтра первого порядка с разными T
        plot_signals(t, u1, T_values[T], 'Фильтр первого порядка', T_values=T_values)

    def special():
        u2 = g + c * np.sin(d * t)
        # Подбираем значения для специального фильтра
        T1 = 0.01
        T2 = 0.3
        T3 = 0.3

        # Применяем специальный фильтр
        num, den = special_filter(T1, T2, T3)
        filtered = signal.filtfilt(num, den, u2)

        # Строим графики для специального фильтра
        plot_signals(t, u2, filtered, 'Специальный фильтр (T1={}, T2={}, T3={})'.format(T1, T2, T3),
                     T_values=[T1, T2, T3])

    #default()
    special()


if __name__ == '__main__':
    main()
