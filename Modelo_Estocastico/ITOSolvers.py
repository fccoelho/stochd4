# -*- coding:utf-8 -*-
u"""
Solvers of ITO SDE equations
Created on 09/03/15
by fccoelho
license: GPL V3 or Later
"""
from random import normalvariate
from math import sqrt
import numpy as np
from matplotlib import pyplot as P
import sympy
from sympy import matrices as M


def EulerMaruyama(xstart, ystart, xfinish, nsteps, f1, f2, params):
    """
    Euler-Maruyama solver
    :param xstart: initial x value
    :param ystart: initial y value
    :param xfinish: final x value
    :param nsteps: number of steps to simulate between `xstart` and `xfinish`
    :param f1: function representing the deterministic part of the model
    :param f2: function representing the deterministic part of the model
    :param params: tuple of parameters to be passed to f1 and f2
    :return:
    """
    sol = [ystart]
    xvals = [xstart]
    h = (xfinish - xstart) / nsteps
    for step in range(nsteps):
        sol.append(sol[-1] + h * f1(sol[-1], *params) + np.sqrt(h) * f2(sol[-1], *params) * normalvariate(0, 1))
        xvals.append(xvals[-1] + h)
    return xvals, sol


def EulerMaruyamaMatrix(xstart, ystart, xfinish, nsteps, mu, B, params, vsymbols):
    """
    Euler-Maruyama solver for multidimensional models. Model is passed in matrix form
    :param xstart: initial x value
    :param ystart: initial y value
    :param xfinish: final x value
    :param nsteps: number of steps to simulate between `xstart` and `xfinish`
    :param mu: Mean vector of the model
    :param B: Diffusion matrix for the model
    :param params: dictionary of parameters to be passed to f1 and f2
    :param vnames: variable symbols in order of the `ystart` vector
    :return:
    """
    transitions = B.shape[1]
    muf = sympy.lambdify(vsymbols, mu)  # Turn array into a function which can be evaluated
    Bf = sympy.lambdify(vsymbols, B)
    sol = sympy.zeros(len(ystart), nsteps)
    xvals = sympy.zeros(nsteps, 1)
    sol[:, 0] = ystart
    xvals[0, 0] = xstart
    h = (xfinish - xstart) / float(nsteps)
    dW = sympy.Matrix(np.random.normal(0, 1, (transitions, 1)) * np.sqrt(h))
    step = 1;
    current_state = sol[:, 0]
    while step < (nsteps+1):
        print(current_state[:4, 0], h)
        if np.any(np.array(current_state) < 0):
            sol = sol[:, :-1]
            break
        args = tuple(current_state)+params
        print(args)
        print(mu[0, 0])
        print(muf(*args))
        sol[:, step] = current_state + (h * np.array(muf(*args))) + np.array(Bf(*args)) * dW
        xvals[step, 0] = xvals[step - 1, 0] + h
        current_state = sol[:, step]
        print(step)
        step += 1
    return xvals, sol


if __name__ == "__main__":
    # Example of simple exponential growth model
    b = 0.3
    d = 0.27
    y0 = 15
    t = np.linspace(0, 150, 500)
    y = y0 + np.exp((b - d) * t)
    P.plot(t, y, label="exact")

    det = lambda y, b, d: (b - d) * y
    stoc = lambda y, b, d: sqrt((b + d) * y)

    x, sol = EulerMaruyama(0, y0, 150, 500, det, stoc, (b, d))
    P.plot(x, sol, label='em')
    P.grid()
    P.show()
