import matplotlib.pyplot as plt
import pydub
import numpy as np


class Plot:

    def __init__(self, x, y, xLabel, yLabel):
        self.x = x
        self.y = y
        self.xLabel = xLabel
        self.yLabel = yLabel


def draw(plots, title, setLimits=True, limits=(2.5, 1.2), T=None):
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

        labelsY.append(yLabel)
        labelsX.append(xLabel)

    plt.xlabel(labelsX[0])
    plt.ylabel(labelsY[0])
    if setLimits:
        plt.xlim(-limits[0], limits[0])
        plt.ylim(-limits[1], limits[1])
    plt.title(title)
    plt.grid(True)
    plt.legend(labelsY)
    plt.show()


def readAudio(f, normalized=False):
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2 ** 15
    else:
        return a.frame_rate, y
