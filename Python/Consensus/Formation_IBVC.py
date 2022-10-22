# -*- coding: utf-8 -*-
"""
    2018
    @author: E Chávez Aparicio (Bloodfield)
    @email: edgar.chavez@cimat.mx
    version: 1.0
    This code cointains a position based control
    of a camera using the Essential Matrix.
"""
import numpy as np
from numpy.linalg import inv, svd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import shutil, os
import csv
import graph as gr
import camera as cm

def ReadF(filename):
    
    listWords = []
    try:
        f =  open(filename)
    except:
        return None
    
    for line in f:
        tmp = line.rstrip()
        tmp = tmp.split(" ")
        tmp = [eval(i) for i in tmp]
        listWords.append( tmp )
    f.close()
    return listWords

def graph(Mat):
    
    np.array()

def main():
    
    
    #   Parameters
    
    n_ag=4 #Number of agents
    n_case = 1  #   Case selector if needed
    directed = False
    n_points = 4 #Number of image points
    depthOp=1 #Depth estimation for interaction matrix, 1-Updated, 2-Initial, 3-Final, 4-Arbitrary fixed, 5-Average
    init_rand =False
    fcntl=1 #1-IBC, 2-HBC
    
    #   Read data
    
    name = 'data/ad_mat_'+str(n_points)
    if directed:
        name += 'd_'
    else:
        name += 'u_'
    name += str(n_case)+'.dat'
    A_ad = ReadF(name)
    if A_ad is None:
        print('File ',name,' does not exist')
        return
    print(A_ad)
    
    G = gr.graph(A_ad, directed)
    #G.plot()
    L = G.laplacian()
    print(L)
    
    [U,S,V]=svd(L)
    lam_n=S[0]
    lam_2=S[-2]
    alpha=2/(lam_n+lam_2)
    A_ds=np.ones(n_ag)-alpha*L
    print(A_ds)
    
    #   READ n_points
    
    #   READ camera init config
    
    
if __name__ ==  "__main__":
    main() 

