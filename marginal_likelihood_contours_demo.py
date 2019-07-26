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
from scipy.special import gamma
import pandas as pd

# specify alpha and beta, alpha = 50, beta = 5 gives dof 100 and variance of 5/50 = 0.1

alpha = 0.2
beta = 0.5
Nos_data = 50000
# Sample x
x = np.random.standard_t(2*alpha, size=Nos_data)

# Define our mean vector, it is 0
u = np.zeros(Nos_data)

# de- normalise x to obtain y
#k = np.sqrt(2*beta)
k = np.sqrt(beta/alpha)
y = k* x + u

# Studentt histogram
size, scale = 1000, 10
commutes = pd.Series(y)

commutes.plot.hist(grid=True, bins=100, rwidth=0.9,
                   color='#607c8e')
plt.title('student t mean 1')
plt.xlabel('counts')
plt.ylabel('x')
plt.grid(axis='y', alpha=0.75)
plt.show()

print('mean=', np.mean(y))
print('variance=', np.var(y))

print(beta/alpha)


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
    for n in range(length):
        Num1 = np.log(gamma((2.*alpha + 1)/2))
        Den1 = np.log(pow(2.*beta*np.pi, 1./2) * gamma(alpha))
        Sum1 = Num1 - Den1
        Sum2 = -1*(2.*alpha + 1)/2 * np.log(1+(1/(2*beta))*pow((y[n]-u[n]), 2))
        d = Sum1 + Sum2
        loglikelihood += d
    return((loglikelihood))
    
    
print(mlikelihood(y, u, 50, 5))


def Max_ab(P, y, u, alpha_values, beta_values, resoln):
    maxP = np.max(P)
    ind = np.unravel_index(np.argmax(P), P.shape)
    assert  P[ind] == maxP, "Index of P did not give max P"
    print('correct index of P found')
    assert  mlikelihood(y, u, alpha_values[ind[1]], beta_values[ind[0]]) == maxP, "Incorrect values to give max P"
    print('Max value of P is', mlikelihood(y, u, alpha_values[ind[1]], beta_values[ind[0]]))
    
    #Scatter plot
    
    a = alpha_values[ind[1]]
    b = beta_values[ind[0]]
    print(len(P[:, ind[1]]), len(beta_values))
    #print(beta_values)
    plt.scatter(beta_values, P[:, ind[1]])
    plt.xlabel('beta')
    plt.ylabel('Log Marginal likelihood')
    plt.grid()
    plt.title('Marginal likelihood along line alpha = %f' %a)
    plt.show()
    
    plt.scatter(alpha_values, P[ind[0]])
    plt.xlabel('alpha')
    plt.ylabel('Log Marginal likelihood')
    plt.grid()
    plt.title('Marginal likelihood along line beta = %f' %b)
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
    
    plt.xlabel('alpha)')
    plt.ylabel('beta')
    plt.grid()
    plt.title('Log Marginal likelihood')
    plt.show()
    
    
    #Contour fill plot
    breaks = np.linspace(minP, maxP, 1000)
    tick = np.linspace(minP, maxP, 11)
    
    PLT2 = plt.contourf((alpha_values), (beta_values), P, breaks) #cmap='seismic')
    plt.colorbar(ticks= tick, orientation='vertical')
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.grid()
    plt.title('Log Marginal likelihood')
    
    maxP = np.max(P)
    ind = np.unravel_index(np.argmax(P), P.shape)
    a = alpha_values[ind[1]]
    b = beta_values[ind[0]]
    
    
    plt.scatter(a, b, c='k')

    plt.show()
    
    return(None)
    
main(y, u, 2, 2, 101)