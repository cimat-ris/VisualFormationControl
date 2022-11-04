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
from numpy.random import rand, randint

#   Image
import cv2
import shutil, os
import csv

#   Custom
import graph as gr
import camera as cm
import initial_params as ip
import controller as ctr
import myplots as mp

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

def plot_3Dcam(ax, camera,
               positionArray,
               init_configuration,
               desired_configuration,
               color,
               camera_scale    = 0.02):
    
    ax.plot(positionArray[0,:],
            positionArray[1,:],
            positionArray[2,:],
            color = color) # Plot camera trajectory
    
    camera.draw_camera(ax, scale=camera_scale, color='red')
    camera.pose(desired_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='brown')
    camera.pose(init_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='blue')
    ax.set_xlabel("$w_x$")
    ax.set_ylabel("$w_y$")
    ax.set_zlabel("$w_z$")
    ax.grid(True)
    
    
def main():
    
    #   Data
    P = np.array(ip.P)      #   Scene points
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
    
    p0 = np.array(ip.p0)    #   init positions
    n_agents = p0.shape[1] #Number of agents
    
    pd = ip.circle(n_agents,0.6,1.2)  #   Desired pose in a circle
    #p0 = pd
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
    
    #   Conectivity graph
    print('plot')
    G.plot()
    
    
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
    
    #   Storage variables
    t_array = np.arange(t,t_end+dt,dt)
    err_array = np.zeros((n_agents,2*n_points,steps))
    U_array = np.zeros((n_agents,6,steps))
    desc_arr = np.zeros((n_agents,2*n_points,steps))
    pos_arr = np.zeros((n_agents,3,steps))
    
    #   LOOP
    for i in range(steps):
        
        print("loop",i)
        
        #   Error:
        
        error = np.zeros((n_agents,2*n_points))
        for j in range(n_agents):
            error[j,:] = agents[j].error
        error = -L @ error
        print(error)
        #   save data
        #print(error)
        err_array[:,:,i] = error
        
        
        ####   Image based formation
        
        #   Get control
        for j in range(n_agents):
            U = agents[j].get_control(error[j,:],G.deg[j],1.0)
            if U is None:
                print("Invalid Ls matrix")
                break
            
            print('U= ',U)
            U_array[j,:,i] = U
            agents[j].update(U,dt,P)
            
            #   save data 
            desc_arr[j,:,i] = agents[j].s_current.T.reshape(2*n_points)
            pos_arr[j,:,i] = agents[j].camera.p
        
        #   Homography based
        
        
        #   Update
        t += dt
        
    ####   Plot
    
    # Colors setup
    
    #        RANDOM X_i
    colors = (randint(0,255,3*max(n_agents,2*n_points))/255.0).reshape((max(n_agents,2*n_points),3))
    
    #   Camera positions (init, end, ref) 
    lfact = 1.1
    mp.plot_position(pos_arr,
                    pd,
                    lfact,
                    colors,
                    name = "Cameras_trayectories",
                    label = "Positions")
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    label = "World setting"
    name = "3Dplot"
    fig.suptitle(label)
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    for i in range(n_agents):
        plot_3Dcam(ax, agents[i].camera,
                pos_arr[i,:,:],
                p0[:,i],
                pd[:,i],
                color = colors[i],
                camera_scale    = 0.02)
    plt.savefig(name+'.png',bbox_inches='tight')
    
    #   Descriptores (init, end, ref) x agente
    for i in range(n_agents):
        mp.plot_descriptors(desc_arr[i,:,:],
                            agents[i].camera.iMsize,
                            agents[i].s_ref,
                            colors,
                            name = "Image_Features_"+str(i),
                            label = "Image Features")
    
    #   Errores 
    for i in range(n_agents):
        mp.plot_time(t_array,
                    err_array[i,:,:],
                    colors,
                    name = "Features_Error_"+str(i),
                    label = "Features Error")
    
    #   Velocidaes x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    U_array[i,:,:],
                    colors,
                    name = "Velocidades_"+str(i),
                    label = "Velocidades",
                    labels = ["X","Y","Z","Wx","Wy","Wz"])
    
    
    
if __name__ ==  "__main__":
    main() 

