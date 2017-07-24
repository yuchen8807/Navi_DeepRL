# -*- coding: utf-8 -*-
'''
implementations for reinforcement learning

produce the minimum-jerk movement

yuchen 2017.02.15
deng@informatik.uni-hamburg.de
'''
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
#from IPython.display import display, Math, Latex
from sympy import symbols, Matrix, latex, Eq, collect, solve, diff, simplify
from sympy.utilities.lambdify import lambdify



def miniJerk_3D(start_position, target_position):
    '''Func: produce a 3D minimum-jerk movement '''
    #---- declare the symbolic variables
    x, xi, xf, y, yi, yf, z, zi, zf, d, t = symbols('x, x_i, x_f, y, y_i, y_f, z, zi, zf,d, t')
    a0, a1, a2, a3, a4, a5 = symbols('a_0:6')
    x = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    #display(Math(latex('x(t)=') + latex(x)))

    #---- define the boundary equations: consider $ t_i=0 $ and  $ d $ for movement duration ($ d=t_f $).
    #The system of equations with the boundary conditions for $ x $:
    s = Matrix([Eq(x.subs(t,0)          , xi),
                Eq(diff(x,t,1).subs(t,0),  0),
                Eq(diff(x,t,2).subs(t,0),  0),
                Eq(x.subs(t,d)          , xf),
                Eq(diff(x,t,1).subs(t,d),  0),
                Eq(diff(x,t,2).subs(t,d),  0)])
    #display(Math(latex(s, mat_str='matrix', mat_delim='[')))

    #----algebraically solve the system of equations
    sol = solve(s, [a0, a1, a2, a3, a4, a5])
    #display(Math(latex(sol)))

    #---- substitute the equation parameters by the solution: x, y, z
    x2 = x.subs(sol)
    x2 = collect(simplify(x2, ratio=1), xf-xi)
    #display(Math(latex('x(t)=') + latex(x2)))
    y2 = x2.subs([(xi, yi), (xf, yf)])
    #display(Math(latex('y(t)=') + latex(y2)))
    z2 = x2.subs([(xi, zi), (xf, zf)])
    #display(Math(latex('z(t)=') + latex(z2)))

    #---- substitute by the numerical values: state_position and target_position , time duration
    x2  = x2.subs([(xi, start_position[0]), (xf, target_position[0]), (d, 1)])
    xfu = lambdify(t, diff(x2, t, 0), 'numpy') #position
    vfu = lambdify(t, diff(x2, t, 1), 'numpy') # velocity
    afu = lambdify(t, diff(x2, t, 2), 'numpy') # accelerate
    jfu = lambdify(t, diff(x2, t, 3), 'numpy') # jerk
    ts = np.arange(0, 1.01, .05) # time step
    position_x = xfu(ts) #now- we only use position
    #print(position_x, 'size', len(position_x))

    y2  = y2.subs([(yi, start_position[1]), (yf, target_position[1]), (d, 1)])
    yfu = lambdify(t, diff(y2, t, 0), 'numpy') #position
    vfu = lambdify(t, diff(y2, t, 1), 'numpy') # velocity
    afu = lambdify(t, diff(y2, t, 2), 'numpy') # accelerate
    jfu = lambdify(t, diff(y2, t, 3), 'numpy') # jerk
    ts = np.arange(0, 1.01, .05) # time step
    position_y = yfu(ts) #now- we only use position
    #print(position_y, 'size', len(position_y))

    z2  = z2.subs([(zi, start_position[2]), (zf, target_position[2]), (d, 1)])
    zfu = lambdify(t, diff(z2, t, 0), 'numpy') #position
    vfu = lambdify(t, diff(z2, t, 1), 'numpy') # velocity
    afu = lambdify(t, diff(z2, t, 2), 'numpy') # accelerate
    jfu = lambdify(t, diff(z2, t, 3), 'numpy') # jerk
    ts = np.arange(0, 1.05, .05) # time step
    position_z = zfu(ts) #now- we only use position
    #print(position_z, 'size', len(position_z))

    return position_x, position_y, position_z
#end of miniJerk_3D mehod

def miniJerk_1D(start_position, target_position):
    '''Func: produce a 3D minimum-jerk movement '''
    #---- declare the symbolic variables
    x, xi, xf, d, t = symbols('x, x_i, x_f,d, t')
    a0, a1, a2, a3, a4, a5 = symbols('a_0:6')
    x = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
    #display(Math(latex('x(t)=') + latex(x)))

    #---- define the boundary equations: consider $ t_i=0 $ and  $ d $ for movement duration ($ d=t_f $).
    #The system of equations with the boundary conditions for $ x $:
    s = Matrix([Eq(x.subs(t,0)          , xi),
                Eq(diff(x,t,1).subs(t,0),  0),
                Eq(diff(x,t,2).subs(t,0),  0),
                Eq(x.subs(t,d)          , xf),
                Eq(diff(x,t,1).subs(t,d),  0),
                Eq(diff(x,t,2).subs(t,d),  0)])
    #display(Math(latex(s, mat_str='matrix', mat_delim='[')))

    #----algebraically solve the system of equations
    sol = solve(s, [a0, a1, a2, a3, a4, a5])
    #display(Math(latex(sol)))

    #---- substitute the equation parameters by the solution: x, y, z
    x2 = x.subs(sol)
    x2 = collect(simplify(x2, ratio=1), xf-xi)

    #---- substitute by the numerical values: state_position and target_position , time duration
    x2  = x2.subs([(xi, start_position), (xf, target_position), (d, 1)])
    xfu = lambdify(t, diff(x2, t, 0), 'numpy') #position
    vfu = lambdify(t, diff(x2, t, 1), 'numpy') # velocity
    afu = lambdify(t, diff(x2, t, 2), 'numpy') # accelerate
    jfu = lambdify(t, diff(x2, t, 3), 'numpy') # jerk
    ts = np.arange(0, 1.01, .05) # time step
    position_x = xfu(ts) #now- we only use position
    return position_x
# end of miniJerk_1D method

if __name__ == '__main__':
    import pandas as pd
    start_position = [0, 0, 0]
    target_position = [1, 2, 3]
    px, py, pz = miniJerk_3D(start_position, target_position)
    #print('position_3D', px, py, pz)
    position_3D = np.vstack((px, py, pz))
    print('position_3D', position_3D)

    columns_name = ['Px', 'Py', 'Pz']
    pd_miniJerk = pd.DataFrame(position_3D.T, index = None, columns = columns_name)
    pd_miniJerk.to_csv('results/miniJerk_movement.csv')

    position_1D = miniJerk_1D(0, 1)
    #print('position_1D', position_1D)
