#!/usr/bin/env python3
"""
Deterministic version of the 4-strain dengue model
with cross-immunity

Created on 29/06/15
by fccoelho
license: GPL V3 or Later
"""

import PyDSTool as dst
import numpy as np
import matplotlib.pyplot as plt

Model4st = dst.args(name='Dengue4')

# Parameters
Model4st.pars = {
    'beta': 400/52,
    'N': 50000,
    'delta': 0.2,  # Cross-immunity protection
    'mu': 1/(70*52),  # Mortality rate
    'sigma': 1/1.5  # recovery rate
}

# Equations
Model4st.varspecs = {
    'S': '-beta * S * (I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
         I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342+\
         I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243+\
         I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) + mu*N - mu*S',
    'I_1': 'beta * S * (I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
          - sigma*I_1 - mu*I_1',
    'I_2': 'beta * S * (I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
          - sigma*I_2 - mu*I_2',
    'I_3': 'beta * S * (I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
          - sigma*I_3 - mu*I_3',
    'I_4': 'beta * S * (I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
           - sigma*I_4 - mu*I_4',
    'R_1': 'sigma*I_1 - beta*delta*R_1*\
           (I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342+\
           I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243+\
           I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_1',
    'R_2': 'sigma*I_2 - beta*delta*R_2*\
           (I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
           I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243+\
           I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_2',
    'R_3': 'sigma*I_3 - beta*delta*R_3*\
           (I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
           I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342+\
           I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_3',
    'R_4': 'sigma*I_4 - beta*delta*R_4*\
           (I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
           I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342+\
           I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243) - mu*R_4',
    'I_12': 'beta*delta*R_1*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_12 - mu*I_12',
    'I_13': 'beta*delta*R_1*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_13 - mu*I_13',
    'I_14': 'beta*delta*R_1*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_14 - mu*I_14',
    'I_21': 'beta*delta*R_2*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_21 - mu*I_21',
    'I_23': 'beta*delta*R_2*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_23 - mu*I_23',
    'I_24': 'beta*delta*R_2*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_24 - mu*I_24',
    'I_31': 'beta*delta*R_3*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_31 - mu*I_31',
    'I_32': 'beta*delta*R_3*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_32 - mu*I_32',
    'I_34': 'beta*delta*R_3*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_34 - mu*I_34',
    'I_41': 'beta*delta*R_4*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_41 - mu*I_41',
    'I_42': 'beta*delta*R_4*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_42 - mu*I_42',
    'I_43': 'beta*delta*R_4*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_43 - mu*I_43',
    'R_12': 'sigma*(I_12+I_21) - beta*delta*\
            R_12*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243+\
            I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_12',
    'R_13': 'sigma*(I_13+I_31) - beta*delta*\
            R_13*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342+\
            I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_13',
    'R_14': 'sigma*(I_14+I_41) - beta*delta*\
            R_14*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342+\
            I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243) - mu*R_14',
    'R_23': 'sigma*(I_23+I_32) - beta*delta*\
            R_23*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
            I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_23',
    'R_24': 'sigma*(I_24+I_42) - beta*delta*\
            R_24*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
            I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243) - mu*R_24',
    'R_34': 'sigma*(I_34+I_43) - beta*delta*\
            R_34*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341+\
            I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342) - mu*R_34',
    'I_231': 'beta*delta*R_23*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_231 - mu*I_231',
    'I_241': 'beta*delta*R_24*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_241 - mu*I_241',
    'I_341': 'beta*delta*R_34*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_341 - mu*I_341',
    'I_132': 'beta*delta*R_13*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_132 - mu*I_132',
    'I_142': 'beta*delta*R_14*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_142 - mu*I_142',
    'I_342': 'beta*delta*R_34*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_342 - mu*I_342',
    'I_123': 'beta*delta*R_12*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_123 - mu*I_123',
    'I_143': 'beta*delta*R_14*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_143 - mu*I_143',
    'I_243': 'beta*delta*R_24*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_243 - mu*I_243',
    'I_124': 'beta*delta*R_12*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_124 - mu*I_124',
    'I_134': 'beta*delta*R_13*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_134 - mu*I_134',
    'I_234': 'beta*delta*R_23*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_234 - mu*I_234',
    'R_123': 'sigma*(I_123+I_132+I_231) - beta*delta*\
            R_123*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234) - mu*R_123',
    'R_124': 'sigma*(I_124+I_241+I_142) - beta*delta*\
            R_124*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243) - mu*R_124',
    'R_134': 'sigma*(I_134+I_341+I_143) - beta*delta*\
            R_134*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342) - mu*R_134',
    'R_234': 'sigma*(I_234+I_342+I_243) - beta*delta*\
            R_234*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342) - mu*R_134',
    'I_1234': 'beta*delta*R_123*(I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234)\
            -sigma*I_1234 - mu*I_1234',
    'I_1243': 'beta*delta*R_124*(I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243)\
            -sigma*I_1243 - mu*I_1243',
    'I_1342': 'beta*delta*R_134*(I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342)\
            -sigma*I_1342 - mu*I_1342',
    'I_2341': 'beta*delta*R_234*(I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341)\
            -sigma*I_2341 - mu*I_2341',
    'R_1234': 'sigma*(I_1234+I_1243+I_1342+I_2341) - mu*R_1234',
    # 'I_a1': 'I_1+I_21+I_31+I_41+I_231+I_241+I_341+I_2341',
    # 'I_a2': 'I_2+I_12+I_32+I_42+I_132+I_142+I_342+I_1342',
    # 'I_a3': 'I_3+I_13+I_23+I_43+I_123+I_143+I_243+I_1243',
    # 'I_a4': 'I_4+I_14+I_24+I_34+I_124+I_134+I_234+I_1234',
    # 'I_all': 'I_a1+I_a2+I_a3+I_a4',
    # 'R_all': 'R_1+R_2+R_3+R_4+R_12+R_13+R_14+R_23+R_24+R_34+R_123+R_124+R_134+R_234+R_1234',
    # 'N': 'S+I_all+R_all'
}

# Initial conditions
Model4st.ics = {'S': 48000, 'I_1': 500, 'I_2': 500, 'I_3': 500, 'I_4': 500, 'R_1': 0, 'R_2': 0, 'R_3': 0, 'R_4': 0, 'I_12': 0, 'I_13': 0, 'I_14': 0, 'I_21': 0, 'I_23': 0, 'I_24': 0, 'I_31': 0, 'I_32': 0, 'I_34': 0, 'I_41': 0, 'I_42': 0, 'I_43': 0, 'R_12': 0, 'R_13': 0, 'R_14': 0, 'R_23': 0, 'R_24': 0, 'R_34': 0, 'I_231': 0, 'I_241': 0, 'I_341': 0, 'I_132': 0, 'I_142': 0, 'I_342': 0, 'I_123': 0, 'I_143': 0, 'I_243':0, 'I_124': 0, 'I_134': 0,
     'I_234': 0, 'R_123': 0, 'R_124': 0, 'R_134': 0, 'R_234': 0, 'I_1234': 0, 'I_1243': 0, 'I_1342': 0, 'I_1342': 0, 'I_2341': 0, 'R_1234': 0,
# 'I_a1': 0, 'I_a2': 0, 'I_a3': 0, 'I_a4': 0, 'I_all': 2000, 'R_all': 0, 'N': 50000
                }

# Simulation
dt = 0.1
tf = 10
Model4st.tdomain = [0, tf]             # set the range of integration.
ode  = dst.Generator.Dopri_ODEsystem(Model4st)
ode.set(algparams={'max_pts': 2000000})
traj = ode.compute('Dengue')              # integrate ODE
pts  = traj.sample(dt=dt)


# PyPlot commands
# plt.plot(pts['I_a3'], label='I_a3');
for k in pts.keys():
    if k =="I_a4": continue
    if k.startswith('I_') and k.endswith('4'):
        plt.plot(pts[k], label=k)
# plt.plot(pts['I_a1'], label='I_a1')
# plt.plot(pts['I_a2'], label='I_a2')
# plt.plot(pts['I_a3'], label='I_a3')
# plt.plot(pts['I_a4'], label='I_a4')
# plt.plot(pts['R_all'], label='R')
plt.xlabel('t');                              # Axes labels
plt.ylabel('individuals');                           # ...
#plt.ylim([0,65]);                                # Range of the y axis
plt.title(ode.name)
plt.legend(loc=0)
plt.grid()
plt.show()
