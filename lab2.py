import numpy as np
from utils import Plot, draw, readAudio
import os


aList = [2, 6, 7]
bList = [1, 5, 9]


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


def f1_f(v, index=0):
    return f3(v, index)


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


def f2_f(v, index=0):
    a = aList[index]
    b = bList[index]

    yf = a * (-2j*np.pi*b*v - np.exp(-2j*np.pi*b*v) + 1) / (2 * np.pi**2 * b * v**2)
    return yf


def f3(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a*np.sinc(b * t)
    return y


def f3_f(v, index=0):
    return f1(v, index)


def f4(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a * np.exp(-b * t**2)
    return y


def f5(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a * np.exp(-b * abs(t))
    return y


def drawFunAndFf(f, ff, t, index, tLim, yLim, label):
    y = f(t, index)

    yf = ff(t, index)
    draw([Plot(t, y, "t", label)],
         label, limits=(tLim, yLim + 0.3))

    label = label + " (Фурье образ)"
    draw([Plot(t, yf, "v", label)],
         label, limits=(tLim, yLim + 0.3))


def drawFirstDefault():
    tLim = 5
    t = np.linspace(-tLim, tLim, 100)

    for i in range(1):
        a = aList[i]
        drawFunAndFf(f1, f1_f, t, i, tLim, a, "Прямоугольная функция")

        drawFunAndFf(f2, f2_f, t, i, tLim, a, "Треугольная функция")

        drawFunAndFf(f3, f3_f, t, i, tLim, a, "Кардинальный синус")

        y = f4(t, i)
        draw([Plot(t, y, "t", "Функция Гаусса")],
             "Функция Гаусса", limits=(tLim, a + 0.3))

        y = f5(t, i)
        draw([Plot(t, y, "t", "Двустороннее затухание")],
             "Двустороннее затухание", limits=(tLim, a + 0.3))


def thirdTask():
    path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\lab2\\audio.mp3"
    print(path)
    y = readAudio(path)

    t = np.linspace(0, len(y), len(y))
    draw([Plot(t, y, "t", "")],
         "Аудиозапись", setLimits=False)

    V = 1000
    v = [i for i in range(0, V)]

    Y = np.zeros(len(v))

    for k in range(len(v)):
        Y[k] = abs(np.trapz(t, y * np.exp(-2j * np.pi * v[k] * t)))

    draw([Plot(v, Y, "frequency", "level")],
         "Карта частот", setLimits=False)


def main():
    drawFirstDefault()
    #thirdTask()


if __name__ == '__main__':
    main()
