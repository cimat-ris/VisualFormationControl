import numpy as np
from numpy import sin, cos
from numpy.linalg import matrix_rank, inv


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import matplotlib.lines as mlines
import imageio


#   Grafica el error y las salidas en x,y
def plot_time(t_array,
              var_array,
              colors,
              name="time_plot", 
              label="Variable",
              labels = None):
    
    n = var_array.shape[0]
    
    fig, ax = plt.subplots()
    fig.suptitle(label)
    
    symbols = []
    for i in range(n):
        ax.plot(t_array,var_array[i,:] , color=colors[i])
        symbols.append(mpatches.Patch(color=colors[i]))
    if not(labels is None):
        fig.legend(symbols,labels, loc=2)
    
    plt.tight_layout()
    plt.savefig(name+'.png',bbox_inches='tight')
    #plt.show()
    
    #   ---
    
