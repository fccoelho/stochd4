# -*- coding:utf-8 -*-
u"""
Implementation of the 4 strain model using the ITô formalism.
Solvers Euler Maruyama and others are use to simulate.
Created on 09/03/15
by fccoelho
license: GPL V3 or Later
"""
import sympy
from sympy import symbols, init_printing, pprint, sqrt
from sympy.matrices import Matrix, zeros
from ITOSolvers import EulerMaruyamaMatrix

init_printing()

# Defining symbolic variables

(
S, I_1, I_2, I_3, I_4, R_1, R_2, R_3, R_4, I_12, I_13, I_14, I_21, I_23, I_24, I_31, I_32, I_34, I_41, I_42, I_43, R_12,
R_13, R_14, R_23, R_24, R_34, I_231, I_241, I_341, I_132, I_142, I_342, I_123, I_143, I_243, I_124, I_134, I_234, R_123,
R_124, R_134, R_234, I_1234, I_1243, I_1342, I_2341, R_1234) = symbols('S I_1 I_2 I_3 I_4 R_1 R_2 R_3 R_4 I_12 I_13 I_14 I_21 I_23 I_24 I_31 I_32 I_34 I_41 I_42 I_43\
    R_12 R_13 R_14 R_23 R_24 R_34 I_231 I_241 I_341 I_132 I_142 I_342 I_123 I_143 I_243 I_124 I_134\
    I_234 R_123 R_124 R_134 R_234 I_1234 I_1243 I_1342 I_2341 R_1234')

N = symbols('N')
I_a1, I_a2, I_a3, I_a4 = symbols('I_a1 I_a2 I_a3 I_a4')
beta, delta, mu, sigma = symbols("beta delta mu sigma")

# Defining the transition probabilities

p = [
    beta * S * I_a1,
    beta * S * I_a2,
    beta * S * I_a3,
    beta * S * I_a4,
    sigma * I_1, sigma * I_2, sigma * I_3, sigma * I_4,  # Primary recoveries
    R_1 * delta * beta * I_a2,
    R_1 * delta * beta * I_a3,
    R_1 * delta * beta * I_a4,  # Secondary infections of R_1
    R_2 * delta * beta * I_a1,
    R_2 * delta * beta * I_a3,
    R_2 * delta * beta * I_a4,  # Secondary infections of R_2
    R_3 * delta * beta * I_a1,
    R_3 * delta * beta * I_a2,
    R_3 * delta * beta * I_a4,  # Secondary infections of R_3
    R_4 * delta * beta * I_a1,
    R_4 * delta * beta * I_a2,
    R_4 * delta * beta * I_a3,  # Secondary infections of R_4
    sigma * I_12, sigma * I_13, sigma * I_14, sigma * I_21, sigma * I_23, sigma * I_24,
    sigma * I_31, sigma * I_32, sigma * I_34, sigma * I_41, sigma * I_42, sigma * I_43,  # Secondary Recoveries
    R_23 * delta * beta * I_a1,
    R_24 * delta * beta * I_a1,
    R_34 * delta * beta * I_a1,  # Tertiary infections by DENV1
    R_13 * delta * beta * I_a2,
    R_14 * delta * beta * I_a2,
    R_34 * delta * beta * I_a2,  # Tertiary infections by DENV2
    R_12 * delta * beta * I_a3,
    R_14 * delta * beta * I_a3,
    R_24 * delta * beta * I_a3,  # Tertiary infections by DENV3
    R_12 * delta * beta * I_a4,
    R_13 * delta * beta * I_a4,
    R_23 * delta * beta * I_a4,  # Tertiary infections by DENV4
    sigma * I_231, sigma * I_241, sigma * I_341,  # From DENV1
    sigma * I_132, sigma * I_142, sigma * I_342,  # From DENV2
    sigma * I_123, sigma * I_143, sigma * I_243,  # From DENV3
    sigma * I_124, sigma * I_134, sigma * I_234,  # From DENV4
    R_123 * delta * beta * (I_4 + I_14 + I_24 + I_34 + I_124 + I_134 + I_234 + I_1234),
    R_124 * delta * beta * I_a3,
    R_134 * delta * beta * I_a2,
    R_234 * delta * beta * I_a1,  # Quaternary Infections
    sigma * I_1234, sigma * I_1243, sigma * I_1342, sigma * I_2341,  # Quaternary Recoveries
    mu * N,  # Births
    mu * S, mu * I_1, mu * I_2, mu * I_3, mu * I_4, mu * R_1, mu * R_2, mu * R_3, mu * R_4,
    mu * I_12, mu * I_13, mu * I_14, mu * I_21, mu * I_23, mu * I_24, mu * I_31, mu * I_32,
    mu * I_34, mu * I_41, mu * I_42, mu * I_43, mu * R_12, mu * R_13, mu * R_14, mu * R_23,
    mu * R_24, mu * R_34, mu * I_231, mu * I_241, mu * I_341, mu * I_132, mu * I_142, mu * I_342,
    mu * I_123, mu * I_143, mu * I_243, mu * I_124, mu * I_134, mu * I_234, mu * R_123, mu * R_124,
    mu * R_134, mu * R_234, mu * I_1234, mu * I_1243, mu * I_1342, mu * I_2341, mu * R_1234,  # Deaths
]

p = Matrix(p)
pprint(p[:8, 0])

# Creating  the changes matrix, Delta(X_i)

changes = zeros(114, 48)  # 114th line is no-event
for i in range(113):
    if i < 4:  # primary infections
        changes[i, 0] = -1
        changes[i, i + 1] = 1
    elif i >= 4 and i < 8:  # primary recoveries
        changes[i, i + 1] = -1
        changes[i, i + 5] = 1
    elif i >= 8 and i < 11:  # Secondary infections R1
        changes[i, 5] = -1
        changes[i, i + 1] = 1
    elif i >= 11 and i < 14:  # Secondary infections R2
        changes[i, 6] = -1
        changes[i, i + 1] = 1
    elif i >= 14 and i < 17:  # Secondary infections R3
        changes[i, 7] = -1
        changes[i, i + 1] = 1
    elif i >= 17 and i < 20:  # Secondary infections R4
        changes[i, 8] = -1
        changes[i, i + 1] = 1
    elif i >= 20 and i < 32:  # Secondary recoveries
        tgts = [21, 22, 23, 21, 24, 25, 22, 24, 26, 23, 25, 26]  # index in X
        changes[i, i - 11] = -1
        changes[i, tgts[i - 20]] = 1
    elif i >= 32 and i < 44:  # Tertiary infections
        srcs = [24, 25, 26, 22, 23, 26, 21, 23, 25, 21, 22, 24]  # index in X
        changes[i, srcs[i - 32]] = -1
        changes[i, i - 5] = 1
    elif i >= 44 and i < 56:  # Tertiary recoveries
        srcs = range(27, 39)
        tgts = [39, 40, 41, 39, 40, 42, 39, 41, 42, 40, 41, 42]  # index in X
        changes[i, srcs[i - 44]] = -1
        changes[i, tgts[i - 44]] = 1
    elif i >= 56 and i < 60:  # Quaternary infections
        changes[i, i - 17] = -1
        changes[i, i - 13] = 1
    elif i >= 60 and i < 64:  # Quaternary recoveries
        changes[i, i - 17] = -1
        changes[i, 47] = 1
    elif i == 64:  # birth
        changes[i, 0] = 1
    elif i >= 65 and i < 113:  # deaths
        changes[i, i - 65] = -1

pprint(changes)
print(changes.shape)

# Computing the Expectation Vector

E = zeros(48, 1)
for i in range(113):
    E += p[i, 0]*changes.T[:, i]

pprint(E)

# Computing the diffusion matrix

B = zeros(48, 113)
for i in range(48):
    for j in range(113):
        #print i,j
        B[i, j] = sqrt(p[j, 0] * changes.T[i, j])

pprint(B)

# Simulating using Euler-Maruyama
N = 500
inits = sympy.Matrix([[10, 50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 48000]])
xvals, sol = EulerMaruyamaMatrix(0, inits, 500, E, B, )
