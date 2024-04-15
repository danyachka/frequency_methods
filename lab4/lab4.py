import numpy as np
from utils import draw, Plot


def fourier_transform_trapezoidal(signal, dt, t):
    N = len(signal)
    w = 2 * np.pi * np.fft.fftfreq(N, dt)
    fourier = np.zeros(N, dtype=complex)

    for i, currentW in enumerate(w):
        fourier[i] = np.trapz(signal * np.exp(-1j * currentW * t), t)

    fourier = fourier / (2 * np.pi)**0.5
    return w, fourier


def fourier_transform_trapezoidal_reverse(fourier, w, t):
    y = np.zeros(len(fourier), dtype=complex)

    for i, currentT in enumerate(t):
        y[i] = np.trapz(fourier * np.exp(1j * currentT * w), w)

    y = y / (2 * np.pi)**0.5
    return y


# Выполним спектральное дифференцирование
def spectral_differentiation_trapezoidal(fourier, w, t):
    #w, fourier = fourier_transform_trapezoidal(signal, dt, t)
    # Производная от sin(t) это cos(t)
    multiplier = (1j * w)
    derivative_fourier = fourier * multiplier

    #derivative_fourier = getDerivative(fourier, w[1] - w[0])

    # derivative_signal = np.fft.ifft(derivative_fourier).real
    derivative = fourier_transform_trapezoidal_reverse(derivative_fourier, w, t)
    #derivative = np.real(ifft(derivative_fourier))
    return derivative


def getDerivative(y, dt) -> np.ndarray:
    dy = np.zeros_like(y)
    for k in range(len(y) - 1):
        dy[k] = (y[k+1] - y[k]) / dt

    return dy


def firstTask():
    limit = 100
    t = np.linspace(-limit, limit, 4000)

    dt = t[1] - t[0]

    y = np.sin(t)

    g = 0.01 * np.random.uniform(low=-0.5, high=0.5, size=len(t)) + y
    draw(Plot(t, g, "t", "g(t)"), "sin(t) и шум", limits=(100, 5))

    dy = getDerivative(y, dt)
    draw(Plot(t, dy, "t", "dy(t)/dt"), "Производная от sin(t)", limits=(100, 5))

    dg = getDerivative(g, dt)
    draw(Plot(t, dg, "t", "dg(t)/dt"), "Производная от sin(t) и шума", limits=(100, 5))

    w, gFurier = fourier_transform_trapezoidal(g, dt, t)
    draw(Plot(w, gFurier, "w", "F[g(t)]"), "F[g(t)]", setLimits=False)

    dy_dt_spec = spectral_differentiation_trapezoidal(gFurier, w, t)
    re_dy_dt_spec = dy_dt_spec.real
    im_dy_dt_spec = dy_dt_spec.imag
    draw(Plot(t, re_dy_dt_spec, "t", "Re F'[g(t)]"), "Re F'[g(t)]", limits=(100, 5))

    draw(Plot(t, im_dy_dt_spec, "t", "Im F'[g(t)]"), "Im F'[g(t)]", limits=(100, 11))


def main():
    firstTask()


if __name__ == '__main__':
    main()
