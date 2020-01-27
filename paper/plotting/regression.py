import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

def get_f(betas):
    def dot(a, b):
        return a*betas[0] + b*betas[1]
    return dot

def  plot_2d_regression(x, y, z, betas, xlabel, ylabel,
                        dest, xstart=0.7, ystart=0.7):
    """
    overlay data on top of linear regression 

    Args:
        x: x axis values of data points
        y: y axis values of data points
        z: z axis values of data points
        betas: column vector of regression coefficients
        dest: where to save figure
        xstart: start value for x-axis
        ystart: start value for y-axis
    """ 
    Xrange = np.linspace(xstart, 1.0, 100)
    Yrange = np.linspace(xstart, 1.0, 100)
    X, Y = np.meshgrid(Xrange, Yrange)
    f = get_f(betas)
    Z = f(X, Y)
    plt.contourf(X, Y, Z, cmap='RdYlGn')
    plt.scatter(x, y, c=z, s=150, cmap='RdYlGn', edgecolors='black')
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(dest)
