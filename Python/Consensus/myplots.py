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
    

def plot_descriptors(descriptors_array,
                     camera_iMsize,
                     s_ref,
                    colors,
                    name="time_plot", 
                    label="Variable"):
    
    n = descriptors_array.shape[0]/2
    #print(n)
    n = int(n)
    fig, ax = plt.subplots()
    fig.suptitle(label)
    plt.xlim([0,camera_iMsize[0]])
    plt.ylim([0,camera_iMsize[1]])
    
    symbols = []
    for i in range(n):
        ax.plot(descriptors_array[2*i,0],descriptors_array[2*i+1,0],
                '*',color=colors[i])
        ax.plot(descriptors_array[2*i,-1],descriptors_array[2*i+1,-1],
                'o',color=colors[i])
        ax.plot(descriptors_array[2*i,:],descriptors_array[2*i+1,:],
                color=colors[i])
        ax.plot(s_ref[0,i],s_ref[1,i],marker='^',color=colors[i])
    
    
    symbols = [mlines.Line2D([0],[0],marker='*',color='k'),
               mlines.Line2D([0],[0],marker='o',color='k'),
               mlines.Line2D([0],[0],marker='^',color='k'),
               mlines.Line2D([0],[0],linestyle='-',color='k')]
    labels = ["Start","End","reference","trayectory"]
    fig.legend(symbols,labels, loc=2)
    
    plt.tight_layout()
    plt.savefig(name+'.png',bbox_inches='tight')
    #plt.show()

def plot_position(position_array,
                    desired_position,
                    lfact,
                    colors,
                    name="Trayectories", 
                    label="Positions"):
    
    n = position_array.shape[0]
    #print(n)
    n = int(n)
    fig, ax = plt.subplots()
    fig.suptitle(label)
    
    x_min = lfact*position_array[:,0,:].min()
    x_max = lfact*position_array[:,0,:].max()
    y_min = lfact*position_array[:,1,:].min()
    y_max = lfact*position_array[:,1,:].max()
    sizes = [min(x_min,y_min),max(x_max,y_max)]
    plt.xlim(sizes)
    plt.ylim(sizes)
    
    symbols = []
    for i in range(n):
        ax.plot(position_array[i,0,0],position_array[i,1,0],
                '*',color=colors[i])
        ax.plot(position_array[i,0,-1],position_array[i,1,-1],
                'o',color=colors[i])
        ax.plot(position_array[i,0,:],position_array[i,1,:],
                color=colors[i])
        ax.plot(desired_position[0,i],desired_position[1,i],
                marker='^',color=colors[i])
    
    
    symbols = [mlines.Line2D([0],[0],marker='*',color='k'),
               mlines.Line2D([0],[0],marker='o',color='k'),
               mlines.Line2D([0],[0],marker='^',color='k'),
               mlines.Line2D([0],[0],linestyle='-',color='k')]
    labels = ["Start","End","reference","trayectory"]
    fig.legend(symbols,labels, loc=2)
    
    plt.tight_layout()
    plt.savefig(name+'.png',bbox_inches='tight')
    #plt.show()
