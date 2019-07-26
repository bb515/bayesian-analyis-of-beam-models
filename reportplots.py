# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 19:41:55 2019

@author: Ben Boys
"""

import gyroid
import marginal_likelihood_contours
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

csfont = {'fontname':'Times New Roman'}

# Get data
y_200 = gyroid.y(200)
y_250 = gyroid.y(250)
y_300 = gyroid.y(300)

# Get FEM model
FEM_200 = gyroid.u(200)
FEM_250 = gyroid.u(250)
FEM_300 = gyroid.u(300)

# Get I beam model
I_200 = gyroid.model(5, 200)
I_250 = gyroid.model(5, 250)
I_300 = gyroid.model(5, 300)

x = gyroid.x()

print(y_200, I_200, FEM_200)


# Scatter plots

y_values = plt.scatter(x, y_200, marker= "+", label='$y$', color= 'k')
I_values = plt.plot(x, I_200, label ='$u_{I-beam}$')
FEM_values = plt.plot(x, FEM_200, label='$u_{FEM}$')
plt.axis([0, 243, -2.0, 0])
plt.xlabel('$x$', **csfont)
plt.ylabel('$u_z$')
plt.legend()
plt.title('Deflections for W = 200N')
plt.show()

y_values = plt.scatter(x, y_250, marker= "+", label='$y$', color= 'k')
I_values = plt.plot(x, I_250, label ='$u_{I-beam}$')
FEM_values = plt.plot(x, FEM_250, label='$u_{FEM}$')
plt.axis([0, 243, -2.5, 0])
plt.xlabel('$x$', **csfont)
plt.ylabel('$u_z$')
plt.legend()
plt.title('Deflections for W = 250N')
plt.show()

y_values = plt.scatter(x, y_300, marker= "+", label='$y$', color= 'k')
I_values = plt.plot(x, I_300, label ='$u_{I-beam}$')
FEM_values = plt.plot(x, FEM_300, label='$u_{FEM}$')
plt.axis([0, 243, -2.9, 0])
plt.xlabel('$x$', **csfont)
plt.ylabel('$u_z$')
plt.legend()
plt.title('Deflections for W = 300N')
plt.show()
