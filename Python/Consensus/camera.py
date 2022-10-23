import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class camera:
    
    
    foc=0.002; #Focal de la camara
    iMsize=[1024, 1024]; #Not working change this
    pPrinc=[iMsize[0]/2.0, iMsize[1]/2.0]; #Not working change this

    xymin=-0.9
    xymax=0.9
    zmin=0.8
    zmax=1.8
    angsmin=-30
    angsmax=30
    Ldown=180*pi/180
    
