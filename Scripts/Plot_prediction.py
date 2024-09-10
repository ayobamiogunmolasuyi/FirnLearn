import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math

def plot_prediction(y1, y2, n_toplot=10**10, title = "Model Evaluation"):
    
    from scipy.stats import gaussian_kde
    from sklearn.metrics import r2_score, mean_squared_error
    
    idxs = np.arange(len(y1))
    np.random.shuffle(idxs)
    y1 = y1*1000
    y2=y2*1000
    y_expected = y1.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y2.reshape(-1)[idxs[:n_toplot]]


    
    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)*100000
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    plt.figure(figsize=(8,8))
    plt.title(title, fontsize=25)
    plt.ylabel('Modeled density (Kgm$^-3$)', fontsize= 18)
    plt.xlabel('Reference density (Kgm$^-3$)', fontsize= 18)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=25, vmin = 0, vmax = 4)
    #plt.clim(0,50)
    plt.tick_params(labelsize=16)
    cbar = plt.colorbar(sc,  fraction=0.046, pad=0.04) 
    cbar.ax.tick_params(labelsize=20)
    
    # Set the ticks to start at the min and end at the max
    #cbar.set_ticklabels([f"{z.min():.2f}", f"{z.max():.2f}"])  # Custom formatting of tick labels

    lineStart = 0
    lineEnd = 1000
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$RMSE=%.4f$' % (math.sqrt(mean_squared_error(y_expected, y_predicted), )),
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
    
    plt.show()
