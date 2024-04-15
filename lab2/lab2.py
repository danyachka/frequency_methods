import numpy as np
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

    y = 2 * a * b * np.sinc(b * w / np.pi) / (2 * np.pi)**0.5
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

    yf = 2 * a * (np.sin(b * w) / (np.pi * w))**2
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

    y = 2**0.5 * a / (b + 2j * np.pi * w)
    return y


def f4_2(t, c, index=1):
    y = f4(t + c, index)
    return y


def f4_2f(w, c, index=1):
    y = f4_f(w, index) * np.exp(-2j * np.pi * w * c)
    return y


def parsevalEqualityCheck(funName, func, fourier_transform, index, interval=(-10, 10), num_points=2000):
    x = np.linspace(interval[0], interval[1], num_points)
    dx = (interval[1] - interval[0]) / num_points

    integral_func_squared = np.trapz(np.abs(func(x, index)) ** 2, dx=dx)

    integral_fourier_squared = np.trapz(np.abs(fourier_transform(x, index)) ** 2, dx=dx)

    parseval_equality = np.isclose(integral_func_squared, integral_fourier_squared, rtol=1e-1)

    print("\n" + "Проверка равенства Парсеваля для " + Fore.GREEN + funName + Fore.RESET)
    print(f'f(x): {Fore.BLUE}{integral_func_squared}{Fore.RESET}')
    print(f'f^(w): {Fore.BLUE}{integral_fourier_squared}{Fore.RESET}')
    if parseval_equality:
        print(f"Равенство {Fore.GREEN}выполняется{Fore.RESET}")
    else:
        print(f"Равенство {Fore.RED}не выполняется{Fore.RESET}")

    return parseval_equality


def drawFunAndFf(f, ff, t, index, tLim, l):
    a = aList[index]
    b = bList[index]

    label = l + f' (a = {a}, b = {b})'

    y = f(t, index)

    yf = ff(t, index)
    draw([Plot(t, y, "t", " f(t)")],
         label + " (f(t))", limits=(tLim, 0))

    draw([Plot(t, yf, "v", " f^(v)")],
         label + " (f^(v))", limits=(tLim, 0))

    parsevalEqualityCheck(l, f, ff, index)


def drawFirstDefault():
    tLim = 5
    t = np.linspace(-tLim, tLim, 1000)

    # for i in range(len(aList)):
    for i in range(3):
        #drawFunAndFf(f1, f1_f, t, i, tLim, "Прямоугольная функция")

        #drawFunAndFf(f2, f2_f, t, i, tLim, "Треугольная функция")

        #drawFunAndFf(f3, f3_f, t, i, tLim, "Кардинальный синус")

        #drawFunAndFf(f4, f4_f, t, i, tLim, "Функция Гаусса")

        drawFunAndFf(f5, f5_f, t, i, tLim, "Двустороннее затухание")


def secondTask():
    tLim = 5
    w = np.linspace(-tLim, tLim, 1000)

    for c in [2, -10, 22]:
        t = np.linspace(-tLim - c, tLim - c, 1000)
        g = f4_2(t, c)

        draw([Plot(t, g, "t", "g(t)")],
             "g(t) = f(t + c), c = " + str(c), setLimits=False)

        gf = f4_2f(w, c)

        re = np.real(gf)
        draw([Plot(w, re, "w", "Re g^(w)")],
             "Вещественная часть, c = " + str(c), setLimits=False)
        im = np.imag(gf)
        draw([Plot(w, im, "w", "Im g^(w)")],
             "Мнимая часть, c = " + str(c), setLimits=False)

        module = abs(gf)
        draw([Plot(w, module, "w", "abs(g^(w))")],
             "Модуль, c = " + str(c), setLimits=False)


def thirdTask():
    path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\lab2\\audio.mp3"
    print(path)
    y = readAudio(path)

    t = np.linspace(0, len(y), len(y))
    draw([Plot(t, y, "t", "")],
         "Аудиозапись, f(t)", setLimits=False)

    V = 1000
    v = [i for i in range(0, V)]

    Y = np.zeros(len(v))

    for k in range(len(v)):
        Y[k] = abs(np.trapz(t, y * np.exp(-2j * np.pi * v[k] * t)))

    draw([Plot(v, Y, "Частоты", "")],
         "Аудиозапись, abs(f^(t))", setLimits=False)

    highest = []
    for i in range(V):
        level = Y[i]
        if level > 3:
            append = True
            for h in highest:
                append = abs(h - i) > 20
                if not append:
                    continue

            if append:
                highest.append(i)

    print(f'Наиболее часто используемые частоты:\n{highest}')


def main():
    drawFirstDefault()
    #secondTask()
    #thirdTask()


if __name__ == '__main__':
    main()
