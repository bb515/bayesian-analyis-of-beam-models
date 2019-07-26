# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:39:26 2019

@author: Ben Boys


Single variate student-T version sampling from numpy student T
Must take products instead of using vector dot products
"""
import numpy as np
import matplotlib.pyplot as plt
from gyroid import *
from scipy.special import gamma

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
    """Inputs:  P the marginal likelihood array 
                y, the data
                u, the mean
                alpha_values, linspace of alpha values the likelihood is calculated over
                beta_values, linspace of beta values the likelihood is calculated over
    
        Outputs:
    Returns the maximum element in P array (the maximum log likelihood), and corresponding
    alpha and beta that produce the value of the maximum log likelihood. Produces a scatter plot
    of log-likelihood versus alpha values and beta values respectively"""
    maxP = np.max(P)
    ind = np.unravel_index(np.argmax(P), P.shape)
    assert  P[ind] == maxP, "Index of P did not give max P"
    print('correct index of P found')
    assert  mlikelihood(y, u, alpha_values[ind[1]], beta_values[ind[0]]) == maxP, "Incorrect values to give max P"
    print('Max value of P is', mlikelihood(y, u, alpha_values[ind[1]], beta_values[ind[0]]))
    
    #Scatter plot
    
    a = alpha_values[ind[1]]
    b = beta_values[ind[0]]
    
    #print(beta_values)
    plt.scatter(beta_values, P[:, ind[1]])
    plt.xlabel(r'$\beta$')
    plt.ylabel('Log Marginal Likelihood')
    plt.grid()
    plt.title(r'Marginal likelihood along line $\alpha$ = %f' %a)
    plt.show()
    
    plt.scatter(alpha_values, P[ind[0]])
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Log Marginal Likelihood')
    plt.grid()
    plt.title(r'Marginal likelihood along line $\beta$ = %f' %b)
    plt.show()
    
    
    return('alpha =', alpha_values[ind[1]], 'beta =', beta_values[ind[0]])
    
def main(y, u, alpha_max, beta_max, resoln):
    """Inputs:  y, the data vector
                u, the mean vector
                alpha_max, max alpha value to be searched over
                beta_max, max beta value to be searched over
                resoln, the num of the numpy linspace
    
        Outputs:
    For a random variable y, and a known mean, returns a contour plot of the likelihood of the parameters alpha and beta
    which define the distribution of the unknown variance. The values of alpha and beta which give the maximum likelihood
    of the data given the mean vector, alpha, beta are found. These can be used to define an inverse-gamma distribution
    which describes the random variable sigma."""
    alpha_values = (np.linspace(1e-2, alpha_max, resoln))
    beta_values = (np.linspace(1e-2, beta_max, resoln))

    #marginal likelihood
    ALPHA_VALUES, BETA_VALUES = np.meshgrid(alpha_values, beta_values)
    P = mlikelihood(y, u, ALPHA_VALUES, BETA_VALUES)
    print(Max_ab(P, y, u, alpha_values, beta_values, resoln))
    
    
    #marginal likelihood
    A = np.ones([resoln, resoln])
    E = np.e * A
    H = np.power(E, P)
    likelihood = np.sum(H)/np.size(H)
    print('likelihood =', likelihood)
    
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
    plt.title('Log Marginal likelihood')
    plt.show()
    
    
    #Contour fill plot
    breaks = np.linspace(minP, maxP, 1000)
    tick = np.linspace(minP, maxP, 11)
    
    PLT2 = plt.contourf((alpha_values),(beta_values), P, breaks)
    plt.colorbar(ticks= tick, orientation='vertical')
    
    plt.xlabel('alpha')
    plt.ylabel('beta')
    plt.grid()
    plt.title('Log Marginal likelihood')
    plt.show()
    
    return(None)

# Test code
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
