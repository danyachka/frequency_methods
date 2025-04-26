from pickletools import float8

import numpy as np
from utils import Plot, draw
from colorama import Fore


a = -1
b = 1

t0 = 0
t1 = 1
t2 = 2

NList = (1, 3, 4, 5, 100)

T_c = 1
R_c = 2
NList_c = (1, 2, 3, 10, 100)

t0_c = -T_c/8
t1_c = T_c/8
t2_c = 3*T_c/8
t3_c = 5*T_c/8
t4_c = 7*T_c/8


# Функция квадратной волны
def f_sq(space):
    result = np.zeros(len(space))
    for i in range(len(space)):
        t = space[i]
        current = t % (t2 - t0)
        if t0 <= current < t1:
            result[i] = a
            continue
        result[i] = b

    return result


def f_c(t):
    return np.cos(t * np.pi) + np.cos(4*t*np.pi)


def f_nc(t):
    return 0.5 * np.sin(t * np.pi) + np.sin(2*t * np.pi) + 0.1


def f_r(t):
    return np.sin((t * np.pi) - 0.8) - np.cos(4*t*np.pi)


def f_complex(space):
    c = np.zeros(len(space), dtype=np.complex128)

    for i in range(len(space)):
        t = space[i]

        if t0_c <= t < t1_c: c[i] = R_c + 8j*t*R_c/T_c
        elif  t1_c <= t < t2_c: c[i] = 2*R_c - 8*R_c*t/T_c + 1j*R_c #(8 - t) + 2j
        elif  t2_c <= t < t3_c: c[i] = -R_c + 1j*(4*R_c - 8*R_c*t/T_c)#(3 - t) + 1j * (0.375 * t - 5)
        elif  t3_c <= t <= t4_c: c[i] = -6*R_c + 8*R_c*t/T_c - 1j*R_c#(t - 5) + 1j * (2 * t - 2)
    return c


def omega(n, T: float=t2-t0):
    return (2*np.pi*n) / T


def calc(t, y, n):
    omg = omega(n)

    an = (2/(t2-t0)) * np.trapezoid(y * np.cos(omg * t), t)
    bn = (2/(t2-t0)) * np.trapezoid(y * np.sin(omg * t), t)

    cn = (1/(t2-t0)) * np.trapezoid(y * (np.cos(-omg * t) + np.sin(omg * t)), t)
    return an, bn, cn, omg


def calc_F(t, y, T, N):
    F = np.zeros_like(t)

    a_0 = (2 / T) * np.trapezoid(y * np.cos(omega(0) * t), t)
    an =  np.array([(2 / T) * np.trapezoid(y * np.cos(omega(n) * t), t) for n in range(1, N + 1)])
    bn =  np.array([(2 / T) * np.trapezoid(y * np.sin(omega(n) * t), t) for n in range(1, N + 1)])

    F += a_0
    for i in range(len(an)):
        n = i + 1
        F += an[i] * np.cos((omega(n) * t)) + bn[i] * np.sin((omega(n) * t))

    if N == 3:
        print(f"{N=}) a_0={a_0}, an={[float(a) for a in an]}, bn={[float(b) for b in bn]}")

    return F, (a_0, an, bn)

def calc_G(t, y, T, N):
    G = 0
    cn = np.array([
        (1 / T) * np.trapezoid(y * np.exp(-1j * omega(n, T) * t), t)
        for n in range(-N, N + 1)
    ])
    for i in range(len(cn)):
        n = i - N
        G += cn[i] * (np.exp(1j * omega(n, T) * t))

    if N == 3:
        print(f'{N=} cn={[complex(cn[i + 1]) for i in range(len(cn) - 2)]}')
    return G, cn

def F_G_N(t, y, label, isSquare=False):
    plots = [Plot(t, y, "t", "f(t)")]
    draw(plots, label, T=t2-t0)

    T = t2 - t0

    for N in NList:
        F, _ = calc_F(t, y, T, N)
        G, _ = calc_G(t, y, T, N)

        draw([(Plot(t, F, "t", "Fn(t) N = " + str(N)))], label,  T=t2-t0)
        draw([Plot(t, G, "t", "Gn(t) N = " + str(N))], label,  T=t2-t0)

    ##draw(plots, label)


def perserval_equality(f, N):
    T = 2*np.pi
    t = np.linspace(-np.pi, np.pi, 1000)

    ft = f(t)
    norm_squared = calc_norm(ft, t)

    omega_n = np.array([omega(n, T) for n in range(N + 1)])

    def an(n):
        return np.trapezoid(ft * np.cos(omega_n[n] * t), t) * 2 / T

    def bn(n):
        return np.trapezoid(ft * np.sin(omega_n[n] * t), t) * 2 / T

    a_list = np.array([an(i) for i in range(N + 1)])
    # print_results('a', a_list, 4)
    b_list = np.array([bn(i) for i in range(1, N + 1)])
    # print_results('b', b_list)

    c_sum = calc_persival_equality_c(ft, T, t, N)

    ab_sum = np.pi * (a_list[0]**2/2 + np.sum(a_list[1:]**2 + b_list**2))

    print(f'Norm squared: {norm_squared:.5f}, \tSum of abs(c(i))^2 = '
          f'{c_sum:.5f}, \tSum of abs(a(i))^2 + abs(b(i))^2 = {ab_sum:.5f}')

def calc_norm(ft, t):
    return np.trapezoid(np.abs(ft) ** 2, t)

def calc_persival_equality_c(ft, T, t, N):
    omega_n_c = [omega(n, T) for n in range(-N, N + 1)]

    def cn(n):
        return (1 / T) * np.trapezoid(ft * np.exp(-1j * omega_n_c[n] * t), t)

    c = np.array([cn(i) for i in range(-N, N + 1)])
    # print_results('c', c)

    # c_sum = 2 * np.pi * np.sum(np.abs(c) ** 2)
    print(2 * np.pi * np.trapezoid(abs(c) ** 2, omega_n_c))
    print(np.trapezoid(abs(c) ** 2, omega_n_c))
    print(np.trapezoid(abs(c) ** 2))
    print(np.sum(abs(c) ** 2))
    return 2 * np.pi * np.trapezoid(abs(c) ** 2, omega_n_c)


def create_default_plot():
    def print_title(l):
        print(Fore.GREEN + l + Fore.RESET)
    t = np.linspace(-1, 1, 1000)

    y = f_sq(t)
    label = "Квадратная функция"
    print_title(label)
    print()
    perserval_equality(f_sq, 100)
    F_G_N(t, y, label)

    # y = f_c(t)
    # label = "Чётная функция"
    # print_title(label)
    # perserval_equality(f_c, 100)
    # F_G_N(t, y, label)
    #
    # y = f_nc(t)
    # label = "Нечётная функция"
    # print_title(label)
    # perserval_equality(f_nc, 100)
    # F_G_N(t, y, label)
    #
    # y = f_r(t)
    # label = "Случайная функция"
    # print_title(label)
    # F_G_N(t, y, label)


def create_complex():
    t = np.linspace(t0_c, t4_c, 10000)

    T = t4_c - t0_c

    c = f_complex(t)
    label = "Комплексная функция"

    l = NList_c[-1]*2 + 1

    for N in NList_c:
        G, cn = calc_G(t, c, T, N)

        if N == 100 or N == 10:
            print(f"2pi*sum(c^2) = {T * np.sum(abs(cn) ** 2)}")

        G_r = np.real(G)
        G_i = np.imag(G)
        draw([Plot(t, G_i, "t", "G_i_n(t) n = " + str(N))], label + " G_i(t)", setLimits=False)
        draw([Plot(t, G_r, "t", "G_r_n(t) n = " + str(N))], label + " G_r(t)", setLimits=False)
        draw([Plot(G_r, G_i, "t", "Gn(t) n = " + str(N))], label, setLimits=False)

    print(f"norm = {calc_norm(c, t)}")

    draw([Plot(np.real(c), np.imag(c), "Re{f(t)}", "Im{f(t)}", style='-', color='orange')], label,
         setLimits=False, show_legend=False)
    draw([Plot(t, np.real(c), "t", "Re{f(t)}")], label, setLimits=False)
    draw([Plot(t, np.imag(c), "t", "Im{f(t)}")], label, setLimits=False)


if __name__ == '__main__':
    # create_complex()
    create_default_plot()
