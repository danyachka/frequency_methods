import numpy as np
import os
import utils
from utils import draw, draw_plots, Plot, FrameInfo


def get_fourier(u, dt):
    N = len(u)
    v = 2 * np.pi * np.fft.fftfreq(N, dt)
    v = np.fft.fftshift(v)
    fourier = np.fft.fftshift(np.fft.fft(u))

    return v, fourier


def get_reversed_fourier(fourier):
    y = np.fft.ifft(np.fft.ifftshift(fourier))
    return y


def main():
    a = 6

    t = np.linspace(-5, 5, 1000)
    dt = t[1] - t[0]

    t1, t2 = -2.3, 1.6
    g = np.where((t >= t1) & (t <= t2), a, 0)

    #draw(Plot(t, g, "t", "g(t)"), "g(t)", limits=(5, 8))

    def first():
        b = 0.6
        u = g + b * (np.random.rand(len(t)) - 0.5)

        v_0, fourier_u = get_fourier(u, dt)

        v0 = 30
        fourier = np.where(abs(v_0) < v0, fourier_u, 0)

        filtered = get_reversed_fourier(fourier)

        plots_0 = [Plot(t, u, "t", "Зашумленный сигнал"),
                   Plot(t, filtered, "t", "Отфильтрованный сигнал")]

        frames = [FrameInfo(f"Отфильтрованный сигнал (v0 = {v0})", "t", "y")]

        v_f, fourier_f = get_fourier(filtered, dt)
        plots_1 = [Plot(v_0, abs(fourier_u), "v", "Модуль Фурье-образа шумного сигнала"),
                   Plot(v_f, abs(fourier_f), "v", "Модуль Фурье-образа фильтрованного сигнала")]
        frames.append(FrameInfo("Модули Фурье-образов", "v", "Мощность"))

        draw_plots([plots_0, plots_1], (2, 1), frames)

    def second():
        c = 0.5
        d = 50
        b = 1
        u = g + b * (np.random.rand(len(t)) - 0.5) + c * np.sin(d * t)

        v_0, fourier_u = get_fourier(u, dt)

        v0 = 30
        fourier = np.where(abs(v_0) < v0, fourier_u, 0)

        filtered = get_reversed_fourier(fourier)

        plots_0 = [Plot(t, u, "t", "Зашумленный сигнал"),
                   Plot(t, filtered, "t", "Отфильтрованный сигнал")]

        frames = [FrameInfo(f"Отфильтрованный сигнал (v0 = {v0})", "t", "y")]

        v_f, fourier_f = get_fourier(filtered, dt)
        plots_1 = [Plot(v_0, abs(fourier_u), "v", "Модуль Фурье-образа шумного сигнала"),
                   Plot(v_f, abs(fourier_f), "v", "Модуль Фурье-образа фильтрованного сигнала")]
        frames.append(FrameInfo("Модули Фурье-образов", "v", "Мощность"))

        draw_plots([plots_0, plots_1], (2, 1), frames)

    def third():
        c = 1
        d = 50
        b = 0.8
        u = g + b * (np.random.rand(len(t)) - 0.5) + c * np.sin(d * t)

        v_0, fourier_u = get_fourier(u, dt)

        v0 = 40
        fourier = np.where(abs(v_0) > v0, fourier_u, 0)

        filtered = get_reversed_fourier(fourier)

        plots_0 = [Plot(t, u, "t", "Зашумленный сигнал"),
                   Plot(t, filtered, "t", "Отфильтрованный сигнал")]

        frames = [FrameInfo(f"Отфильтрованный сигнал (v0 = {v0})", "t", "y")]

        v_f, fourier_f = get_fourier(filtered, dt)
        plots_1 = [Plot(v_0, abs(fourier_u), "v", "Модуль Фурье-образа шумного сигнала"),
                   Plot(v_f, abs(fourier_f), "v", "Модуль Фурье-образа фильтрованного сигнала")]
        frames.append(FrameInfo("Модули Фурье-образов", "v", "Мощность"))

        draw_plots([plots_0, plots_1], (2, 1), frames)


    # first()
    # second()
    third()


def clearHighest(fourier, level):
    for i in range(len(fourier)):
        if fourier[i] < level:
            fourier[i] = 0

    return fourier


def task_2():
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\data\\lab3\\audio.wav"
    print(path)
    y, sr = utils.readAudio(path)

    dt = 1/sr
    T = dt * len(y)
    t = np.linspace(0, T-dt, len(y))

    soundPlot = Plot(t, y, "", "Оригинальная дорожка")

    v, fourier = get_fourier(y, dt)
    fourier_plot = Plot(v, fourier.copy(), "", "Фурье-образ оригинальной дорожки")

    # clearing
    v_abs = abs(v)
    cleared_fourier = np.where((v_abs > 1960), fourier, 0)
    cleared_fourier = np.where(abs(cleared_fourier.real) < 1000, cleared_fourier, 0)

    cleared_fourier_plot = Plot(v, cleared_fourier, "", "Отчищенный Фурье-образ", alpha=0.6)

    result = get_reversed_fourier(cleared_fourier)
    result[:350] = 0
    result[len(result) - 350:] = 0
    cleared_sound_plot = Plot(t, result, "", "Отчищенная дорожка")

    frames = [FrameInfo("График аудио", "Время", "Амплитуда"),
              FrameInfo("Модуль Фурье-образа", "Частота", "Мощность",
                        xLimits=(0, 2000))]
    draw_plots([[soundPlot, cleared_sound_plot],
                [fourier_plot, cleared_fourier_plot]], (2, 1), frames)

    utils.saveWavAudio(result, sr, "result.wav")


if __name__ == '__main__':
    # main()
    task_2()
