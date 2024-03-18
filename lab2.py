import numpy as np
from utils import Plot, draw, readAudio


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


def f3(t, index=0):
    a = aList[index]
    b = bList[index]

    y = a*np.sinc(b * t)
    return y


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


def drawFirstDefault():
    tLim = 5
    t = np.linspace(-tLim, tLim, 100)

    for i in range(1):
        a = aList[i]
        y = f1(t, i)
        draw([Plot(t, y, "t", "Прямоугольная функция")],
             "Прямоугольная функция", limits=(tLim, a + 0.3))

        y = f2(t, i)
        draw([Plot(t, y, "t", "Треугольная функция")],
             "Треугольная функция", limits=(tLim, a + 0.3))

        y = f3(t, i)
        draw([Plot(t, y, "t", "Кардинальный синус")],
             "Кардинальный синус", limits=(tLim, a + 0.3))

        y = f4(t, i)
        draw([Plot(t, y, "t", "Функция Гаусса")],
             "Функция Гаусса", limits=(tLim, a + 0.3))

        y = f5(t, i)
        draw([Plot(t, y, "t", "Двустороннее затухание")],
             "Двустороннее затухание", limits=(tLim, a + 0.3))


def thirdTask():
    y = readAudio("data\\lab2\\audio.mp3")

    t = np.linspace(0, len(y), len(y))
    draw([Plot(t, y, "t", "")],
         "Аудиозапись", setLimits=False)


def main():
    # drawFirstDefault()
    thirdTask()


if __name__ == '__main__':
    main()
