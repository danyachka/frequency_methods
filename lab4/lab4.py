import numpy as np
from utils import draw, Plot


def fourier_transform_trapezoidal(signal, dt, t):
    N = len(signal)
    v = 2 * np.pi * np.fft.fftfreq(N, dt)
    v = np.fft.fftshift(v)
    fourier = np.zeros(N, dtype=complex)

    for i, currentW in enumerate(v):
        fourier[i] = np.trapz(signal * np.exp(-2j * np.pi * currentW * t), t)

    return v, fourier


def fourier_transform_trapezoidal_reverse(fourier, v, t):
    y = np.zeros(len(fourier), dtype=complex)

    for i, currentT in enumerate(t):
        y[i] = np.trapz(fourier * np.exp(2j * np.pi * currentT * v), v)

    return y


# Выполним спектральное дифференцирование
def spectral_differentiation_trapezoidal(fourier, v, t):
    multiplier = (2j * np.pi * v)
    derivative_fourier = fourier * multiplier

    derivative = fourier_transform_trapezoidal_reverse(derivative_fourier, v, t)
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
    draw(Plot(t, y, "t", "y(t) = sin(t)"), "sin(t)", limits=(4, 4))
    draw(Plot(t, np.cos(t), "t", "y'(t) = cos(t)"), "cos(t)", limits=(4, 4))

    g = np.random.uniform(low=-0.3, high=0.3, size=len(t)) + y
    draw(Plot(t, g, "t", "g(t)"), "sin(t) и шум", limits=(4, 4))

    dy = getDerivative(y, dt)
    draw(Plot(t, dy, "t", "dy(t)/dt"), "Численная производная от sin(t)", limits=(4, 4))

    dg = getDerivative(g, dt)
    draw(Plot(t, dg, "t", "dg(t)/dt"), "Численная производная от sin(t) и шума", setLimits=False)

    w, gFurier = fourier_transform_trapezoidal(g, dt, t)
    draw(Plot(w, gFurier.real, "w", "Фурье-образ (Re)"), "Фурье-образ (Re)", setLimits=False)
    draw(Plot(w, gFurier.imag, "w", "Фурье-образ (Im)"), "Фурье-образ (Im)", setLimits=False)

    dy_dt_spec = spectral_differentiation_trapezoidal(gFurier, w, t)
    re_dy_dt_spec = dy_dt_spec.real
    im_dy_dt_spec = dy_dt_spec.imag
    draw(Plot(t, re_dy_dt_spec / 2000, "t", "Спектральная производная (Re / 2000)"),
         "Спектральная производная (Re / 2000)", limits=(4, 4))
    draw(Plot(t, re_dy_dt_spec, "t", "Спектральная производная(Re)"), "Спектральная производная (Re)", setLimits=False, limits=(100, 5))

    draw(Plot(t, im_dy_dt_spec, "t", "Спектральная производная (Im)"), "Спектральная производная (Im)", setLimits=False, limits=(100, 11))


def main():
    firstTask()


if __name__ == '__main__':
    main()
