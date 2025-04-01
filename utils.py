import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import numpy as np
from scipy.io.wavfile import write


class Plot:

    def __init__(self, x, y, xLabel="", yLabel="", color=None, alpha=1):
        self.x = x
        self.y = y
        self.xLabel = xLabel
        self.yLabel = yLabel
        self.color = color
        self.alpha = alpha


class FrameInfo:

    def __init__(self, title, x, y, xLimits=None, yLimits=None):
        self.title = title
        self.x = x
        self.y = y
        self.xLimits = xLimits
        self.yLimits = yLimits


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


def drawPlots(plotsLists: list[list[Plot]], size, frameInfoLists: list[FrameInfo]):
    colors = ["r", "b", "g", "p"]
    plt.figure(figsize=(12, 12))

    for i, plots in enumerate(plotsLists):
        frameInfo = frameInfoLists[i]

        plt.subplot(size[0], size[1], i + 1)

        for j, plot in enumerate(plots):
            color = plot.color
            if color is None:
                color = colors[j]

            plt.plot(plot.x, plot.y, label=plot.yLabel, color=color, alpha=plot.alpha)

        plt.xlabel(frameInfo.x)
        plt.ylabel(frameInfo.y)
        plt.title(frameInfo.title)

        if frameInfo.xLimits is not None:
            plt.xlim(frameInfo.xLimits[0], frameInfo.xLimits[1])

        if frameInfo.yLimits is not None:
            plt.ylim(frameInfo.yLimits[0], frameInfo.yLimits[1])

        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.show()


def readAudio(f):
    y, sr = librosa.load(f)
    y = librosa.to_mono(y)
    return y, sr


def saveWavAudio(data, sr, path):
    path = Path(__file__).absolute().parent.joinpath('result').joinpath(path)

    scaled_data = np.int16(data * 32767)

    write(path, sr, scaled_data)
