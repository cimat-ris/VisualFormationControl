# -*- coding: utf-8 -*-
"""
    2018
    @author: E Chávez Aparicio (Bloodfield)
    @email: edgar.chavez@cimat.mx
    version: 1.0
    This code cointains a visual based control
    of a camera using the descriptors
"""

#   Math
import numpy as np
from numpy.linalg import inv, svd
from numpy import sin, cos, pi

#   Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#   Image
import cv2
import shutil, os
import csv

#   Custom
import graph as gr
import camera as cm
import initial_params as ip
import controller as ctr

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

def main():
    
    #   Data
    P = np.array(ip.P)      #   Scene points
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.zeros((1,n_points))] # change to homogeneous
    
    p0 = np.array(ip.p0)    #   init positions
    n_agents = p0.shape[1] #Number of agents
    
    pd = ip.circle(n_agents,0.6,1.2)  #   Desired pose in a circle
    
    #   Parameters
    
    case_n = 1  #   Case selector if needed
    directed = False
    depthOp=1 #Depth estimation for interaction matrix, 1-Updated, 2-Initial, 3-Final, 4-Arbitrary fixed, 5-Average
    case_controlable=1 #1-All (6), 2-Horizontal (4)
    
    #   Random inital positions?
    init_rand =False
    #   If True:
    xymin=-0.9
    xymax=0.9
    zmin=0.8
    zmax=1.8
    angsmin=-30
    angsmax=30
    
    #   Read more data
    
    name = 'data/ad_mat_'+str(n_agents)
    if directed:
        name += 'd_'
    else:
        name += 'u_'
    name += str(case_n)+'.dat'
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
    A_ds=np.ones(n_agents)-alpha*L
    print(A_ds)
    
    agents = []
    for i in range(n_agents):
        cam = cm.camera()
        agents.append(ctr.agent(cam,pd[:,i],p0[:,i],P))
        
    #   TODO: set references X,p
    
    
        
    #   TODO: verify points in FOV
    
    #   TODO: Z estimation
    
    #   INIT LOOP
    
    t=0.0
    dt = 0.05
    t_end = 10.0
    steps = int((t_end-t)/dt + 1.0)
    lamb = 1.5*np.ones(6)
    
    d_s_norm=np.zeros((2*n_points,n_agents));
    t_array = np.arange(t,t_end,dt)
    
    #   Storage variables
    
    err_array = np.zeros((n_points,steps))
    
    #   LOOP
    for i in range(steps):
        
        print("loop")
        
        #   Error:
        
        error = np.zeros((n_agents,2*n_points))
        for j in range(n_agents):
            error[i,:] = agents[j].error
        
        error = -L @ error
        
        print(error)
        ####   Image based formation
        
        #   Get control
        for j in range(n_agents):
            U = agents[j].get_control(error[j,:],G.deg[j])
            if U is None:
                print("Invalid Ls matrix")
                return None
            agents[j].update(U,dt,P)
        
        #   Homography based
        
        
        #   Update
        t += dt
        
    
if __name__ ==  "__main__":
    main() 

