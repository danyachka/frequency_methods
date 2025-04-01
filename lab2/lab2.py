from operator import truediv
from signal import signal

import numpy as np
from numpy import ndarray
from pydub.pyaudioop import reverse

from utils import Plot, draw, readAudio
import os
from colorama import Fore, Style


aList = [2, 6, 11]
bList = [1, 7, 9]


def f1(ts, index=0):
    a = aList[index]
    b = bList[index]

    y = np.zeros(len(ts))
    for i in range(len(ts)):
        t = ts[i]
        if abs(t) <= b:
            y[i] = a
        else:
            y[i] = 0

    return y


def f1_f(w, index=0):
    a = aList[index]
    b = bList[index]

    # y = 2 * a * b * np.sinc(b * w / np.pi) / (2 * np.pi)**0.5
    # y = a*b*2**0.5 * np.sinc(b*w)
    y = 2*a/((2 * np.pi)**0.5 * w) * np.sin(b*w)
    return y


def f2(ts, index=0):
    a = aList[index]
    b = bList[index]

    y = np.zeros(len(ts))
    for i in range(len(ts)):
        t = ts[i]
        if abs(t) <= b:
            y[i] = a - abs(a * t / b)
        else:
            y[i] = 0

    return y


def f2_f(w, index=0):
    a = aList[index]
    b = bList[index]

    # yf = 2 * a * (np.sin(b * w) / (np.pi * w))**2
    yf = a*2**0.5/(b * w**2 * np.pi**0.5) * (1 - np.cos(w*b))
    return yf


def f3(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a*np.sinc(b * t)
    return y


def f3_f(w, index=0):
    a = aList[index]
    b = bList[index]

    y = np.zeros(len(w))
    for i in range(len(w)):
        t = w[i]
        if abs(t) <= b:
            y[i] = a / (b * 2**0.5)
        else:
            y[i] = 0

    return y


def f4(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a * np.exp(-b * t**2)
    return y


def f4_f(w, index=0):
    a = aList[index]
    b = bList[index]

    y = a * (np.pi / b)**0.5 * np.exp(- (w ** 2 * np.pi ** 2) / b)
    return y


def f5(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a * np.exp(-b * abs(t))
    return y


def f5_f(w, index=0):
    a = aList[index]
    b = bList[index]

    y = (a*b*2**0.5) / (np.pi**0.5 * (w**2 + b**2))
    return y


def f4_2(t, c, index=1):
    y = f4(t + c, index)
    return y


def f4_2f(w, c, index=1):
    y = np.exp(-1j * w * c)/np.sqrt(2*np.pi) * f4_f(w, index)
    return y


def parseval_equality_check(label, y, yf, t):
    integral_func_squared = np.trapezoid(np.abs(y) ** 2, t)

    integral_fourier_squared = np.trapezoid(np.abs(yf) ** 2, t)

    parseval_equality = np.isclose(integral_func_squared, integral_fourier_squared, rtol=1e-1)

    print("\n" + "Проверка равенства Парсеваля для " + Fore.GREEN + label + Fore.RESET)
    print(f'f(x): {Fore.BLUE}{integral_func_squared: .3f}{Fore.RESET}')
    print(f'f^(w): {Fore.BLUE}{integral_fourier_squared: .3f}{Fore.RESET}')
    if parseval_equality:
        print(f"Равенство {Fore.GREEN}выполняется{Fore.RESET}")
    else:
        print(f"Равенство {Fore.RED}не выполняется{Fore.RESET}")

    return parseval_equality


def draw_func_and_ff(f, ff, t, index, tLim, l):
    a = aList[index]
    b = bList[index]

    label = l + f' (a = {a}, b = {b})'

    y = f(t, index)

    yf = ff(t, index)
    draw([Plot(t, y, "t", " f(t)")],
         label + " (f(t))", limits=(tLim, 0))

    draw([Plot(t, yf, "w", " f^(w)")],
         label + " (f^(w))", limits=(tLim, 0))

    parseval_equality_check(f'{l}({a=}, {b=})', y, yf, t)


def draw_first_default():
    tLim = 5
    t = np.linspace(-tLim, tLim, 1000)

    # for i in range(len(aList)):
    for i in range(3):
        # draw_func_and_ff(f1, f1_f, t, i, tLim, "Прямоугольная функция")

        # draw_func_and_ff(f2, f2_f, t, i, tLim, "Треугольная функция")

        # draw_func_and_ff(f3, f3_f, t, i, tLim, "Кардинальный синус")

        # draw_func_and_ff(f4, f4_f, t, i, tLim, "Функция Гаусса")

        draw_func_and_ff(f5, f5_f, t, i, tLim, "Двустороннее затухание")


def second_task():
    tLim = 5
    w = np.linspace(-tLim, tLim, 1000)

    for c in [2, -10, 22]:
        t = np.linspace(-tLim - c, tLim - c, 1000)
        g = f4_2(t, c)

        draw([Plot(t, g, "t", "g(t)")],
             "g(t) = f(t + c), c = " + str(c), setLimits=False)

        gf = f4_2f(w, c)

        re = np.real(gf)
        im = np.imag(gf)
        draw([Plot(w, im, "w", "Im g^(w)"), Plot(w, re, "w", "Re g^(w)")],
             "c = " + str(c), setLimits=False)

        module = abs(gf)
        draw([Plot(w, module, "w", "abs(g^(w))")],
             "Модуль, c = " + str(c), setLimits=False)


def third_task():
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\data\\lab2\\audio.mp3"
    print(path)
    y, _ = readAudio(path)

    t = np.linspace(0, len(y), len(y))
    draw([Plot(t, y, "t", None)],
         "Аудиозапись, f(t)", setLimits=False)

    V = 1000
    s: ndarray = np.array([
        [v, abs(np.trapezoid(t, y * np.exp(-2j * np.pi * v * t)))]
        for v in range(0, V)
    ])

    draw([Plot(s[:, 0], s[:, 1], "Частоты", None)],
         "Аудиозапись, abs(f^(v))", setLimits=False)

    highest = sorted(s, key=lambda x: x[1])
    string = "\n".join([f"v = {v[0]: .0f}, {v[1]: .3f}" for v in highest])
    print(f'Наиболее часто используемые частоты:\n{string}')


def main():
    # draw_first_default()
    # second_task()
    third_task()


if __name__ == '__main__':
    main()
