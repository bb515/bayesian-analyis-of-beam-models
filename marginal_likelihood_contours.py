# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:39:26 2019

@author: Ben Boys


Single variate student-T version sampling from numpy student T
Must take products instead of using vector dot products
"""
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
from gyroid import *
from scipy.special import gamma
import pandas as pd

def mlikelihood(y, u, alpha, beta):
    """Inputs:  y, the data
                u, the mean
                dof, the degrees of freedom
                var, the variance
    
        Outputs:
    This returns the singlevariate studentt likelihood of a random vector
    variable y, with mean u and hyper-parameters degree of freedom and variance.
    It is assumed that the random variables are conditionally independent,
    i.e. the covariance is variance multiplied by the identity matrix."""
    
    length = len(y)
    
    loglikelihood = 0.
    #here are different parts of the function which are split up for readability
    Num1 = np.log(gamma((2.*alpha + 1)/2))
    Den1 = np.log(pow(2.*beta*np.pi, 1./2) * gamma(alpha))
    Num2 = -1*(2.*alpha + 1)/2
    Const1 = (1/(2*beta))
    Sum1 = Num1 - Den1
    for n in range(length):
        Sum2 =  Num2 * np.log(1+Const1*pow((y[n]-u[n]), 2))
        d = Sum1 + Sum2
        loglikelihood += d
    return((loglikelihood))


def Max_ab(P, y, u, alpha_values, beta_values, resoln):
    maxP = np.max(P)
    ind = np.unravel_index(np.argmax(P), P.shape)
    assert  P[ind] == maxP, "Index of P did not give max P"
    print('correct index of P found')
    assert  mlikelihood(y, u, alpha_values[ind[1]], beta_values[ind[0]]) == maxP, "Incorrect values to give max P"
    print('Max value of P is', mlikelihood(y, u, alpha_values[ind[1]], beta_values[ind[0]]))
    
    #Scatter plot
    
    print(len(P[:, ind[1]]), len(beta_values))
    #print(beta_values)
    plt.scatter(beta_values, P[:, ind[1]])
    plt.xlabel('beta')
    plt.ylabel('Marginal likelihood for a given beta')
    plt.grid()
    plt.title('Marginal likelihood')
    plt.show()
    
    plt.scatter(alpha_values, P[ind[0]])
    plt.xlabel('alpha')
    plt.ylabel('Marginal likelihood for a given beta')
    plt.grid()
    plt.title('Marginal likelihood')
    plt.show()
    
    
    return('alpha =', alpha_values[ind[1]], 'beta =', beta_values[ind[0]])
    
def main(y, u, alpha_max, beta_max, resoln):
    
    
    alpha_values = (np.linspace(1e-2, alpha_max, resoln))
    beta_values = (np.linspace(1e-2, beta_max, resoln))

    #marginal likelihood
    ALPHA_VALUES, BETA_VALUES = np.meshgrid(alpha_values, beta_values)
    P = mlikelihood(y, u, ALPHA_VALUES, BETA_VALUES)
    print(Max_ab(P, y, u, alpha_values, beta_values, resoln))
    
    #print(Min_ab(P, dof_values, var_values, data, resoln))

    # Plots
    plt.figure()
    
    # Contour plot
    minP = np.min(P)
    maxP = np.max(P)
    breaks = np.linspace(minP, maxP, 11)

    PLT1 =  plt.contour(alpha_values, beta_values, P,
                        breaks,
                        cmap='seismic'
                        )

    plt.colorbar(ticks=breaks, orientation='vertical')
    plt.clabel(PLT1, inline = 1, fontsize= 10)
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.grid()
    plt.title('Marginal likelihood')
    plt.show()
    
    
    #Contour fill plot
    breaks = np.linspace(minP, maxP, 1000)
    tick = np.linspace(minP, maxP, 11)
    
    PLT2 = plt.contourf(np.log(alpha_values),np.log(beta_values), P, breaks, cmap='seismic')
    plt.colorbar(ticks= tick, orientation='vertical')
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.grid()
    plt.title('Marginal likelihood')
    plt.show()
    
    return(None)




# Get data
y_200 = y(200)
y_250 = y(250)
y_300 = y(300)

# Get FEM model
FEM_200 = u(200)
FEM_250 = u(250)
FEM_300 = u(300)

# Get I beam model
I_200 = model(5, 200)
I_250 = model(5, 250)
I_300 = model(5, 300)


main(y_200, FEM_200, 100, 60, 1000)
print('_____________________________________________________FEM 200 ABOVE')
main(y_200, I_200, 100, 60, 1000)
print('_____________________________________________________I 200 ABOVE')
main(y_250, FEM_250, 100, 60, 1000)
print('_____________________________________________________FEM 250 ABOVE')
main(y_250, I_250, 100, 60, 1000)
print('_____________________________________________________I 250 ABOVE')
main(y_300, FEM_300, 100, 60, 1000)
print('_____________________________________________________FEM 300 ABOVE')
main(y_300, I_300, 100, 60, 1000)
print('_____________________________________________________I 300 ABOVE')


