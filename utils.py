import matplotlib.pyplot as plt
import librosa
import pydub
import numpy as np


class Plot:

    def __init__(self, x, y, xLabel, yLabel):
        self.x = x
        self.y = y
        self.xLabel = xLabel
        self.yLabel = yLabel


def draw(plots: Plot | list[Plot], title: str, setLimits=True, limits=(2.5, 1.2), T=None):
    if isinstance(plots, Plot):
        plots = [plots]
    if len(plots) == 0:
        raise "Zero plots 've been given"
    ax = plt.subplot(1, 1, 1)
    colors = ["r", "b", "g", "p"]
    labelsY = []
    labelsX = []

    isPeriodic = T is not None

    for i in range(0, len(plots)):
        plot, color = plots[i], colors[i % len(colors)]
        x, y, xLabel, yLabel = plot.x, plot.y, plot.xLabel, plot.yLabel

        ax.plot(x, y, "-")
        if isPeriodic:
            ax.plot(x - T, y, "-")
            ax.plot(x + T, y, "-")

        if yLabel is not None:
            labelsY.append(yLabel)
        labelsX.append(xLabel)

    plt.xlabel(labelsX[0])
    if len(labelsY) != 0:
        plt.ylabel(labelsY[0])
        plt.legend(labelsY)

    if setLimits:
        if limits[0] != 0:
            plt.xlim(-limits[0], limits[0])
        if limits[1] != 0:
            plt.ylim(-limits[1], limits[1])
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def readAudio(f):
    y, sr = librosa.load(f)
    y = librosa.to_mono(y)
    return y
