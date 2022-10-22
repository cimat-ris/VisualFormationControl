import numpy as np
from numpy import sin, cos
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class camera:
    
    
    self.foc=0.002; #Focal de la camara
    self.iMsize=[1024, 1024]; #Not working change this
    self.pPrinc=[iMsize[0]/2.0, iMsize[1]/2.0]; #Not working change this

    self.xymin=-0.9
    self.xymax=0.9
    self.zmin=0.8
    self.zmax=1.8
    self.angsmin=-30
    self.angsmax=30
    self.Ldown=180*pi/180
    
