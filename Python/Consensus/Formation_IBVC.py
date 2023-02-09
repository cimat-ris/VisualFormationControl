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
from scipy.optimize import minimize_scalar

#   Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.random import rand, randint

#   Image
#import cv2
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

def plot_3Dcam_end(ax, camera,
               desired_configuration,
               color,
               camera_scale    = 0.02):
    
    
    camera.draw_camera(ax, scale=camera_scale, color='red')
    camera.pose(desired_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='brown')
    ax.set_xlabel("$w_x$")
    ax.set_ylabel("$w_y$")
    ax.set_zlabel("$w_z$")
    ax.grid(True)

def Z_select(depthOp, agent, P, Z_set, p0, pd, j):
    Z = np.ones((1,P.shape[1]))
    if depthOp ==1:
        #M = np.c_[ agent.camera.R, -agent.camera.R @ agent.camera.p ]
        #Z = M @ P
        #Z = Z[2,:]
        Z = agent.camera.p[2]*Z
        Z = Z-P[2,:]
    elif depthOp ==2: # 
        Z = p0[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 3: # distancia al valor inicial
        Z = pd[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 4: # fijo al valor de Z_set
        Z = Z*Z_set
    elif depthOp == 5: # equitativo, = promedio
        tmp = agent.camera.p[2]-np.mean(P[2,:])
        Z = Z*tmp
    else:
        print("Invalid depthOp")
        return None
    return Z


#   Error if state calculation
#   It assumes that the states in the array are ordered and are the same
#   regresa un error de traslación y orientación
def error_state(reference,  agents,colors, name):
    
    n = reference.shape[1]
    state = np.zeros((3,n))
    for i in range(len(agents)):
        state[:,i] = agents[i].camera.p
    
    #   Obten centroide
    centroide_ref = reference[:3,:].sum(axis=1)/n
    centroide_state = state.sum(axis=1)/n
    
    #   Centrar elementos
    new_reference = reference.copy()
    #new_reference[:3,:] = new_reference[:3,:] - centroide_ref
    new_reference[0] -= centroide_ref[0]
    new_reference[1] -= centroide_ref[1]
    new_reference[2] -= centroide_ref[2]
    new_state = state.copy()
    #new_state[:3,:] = new_state[:3,:] - centroide_state
    new_state[0] -= centroide_state[0]
    new_state[1] -= centroide_state[1]
    new_state[2] -= centroide_state[2]
    
    #   Aplicar rotación promedio
    theta = np.arctan2(new_reference[1,:],new_reference[0,:])
    theta -= np.arctan2(new_state[1,:],new_state[0,:])
    theta = theta.mean()
    
    ca = cos(theta)
    sa = sin(theta)
    R = np.array([[ ca, -sa],
                  [ sa,  ca]])
    new_state[:2,:] = R@new_state[:2,:]
    
    #       Updtate normalized and oriented agents
    for i in range(len(agents)):
        new_p = np.r_[new_state[:,i] ,agents[i].camera.roll,agents[i].camera.pitch,agents[i].camera.yaw + theta]
        agents[i].camera.pose(new_p)
        
    #   normalizar referencia
    new_reference[:3,:] /= np.linalg.norm(new_reference[:3,:],axis=0).mean()
    
    #   Aplicar minimización de radio (como lidiar con mínimos múltiples)
    f = lambda r : np.linalg.norm(new_reference[:3,:] - r*new_state[:3,:],axis = 0).sum()
    r_state = minimize_scalar(f, method='brent')
    t_err = f(r_state.x)
    
    #   Plot
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(n):
        agents[i].camera.draw_camera(ax, scale=0.2, color='red')
        agents[i].camera.pose(new_reference[:,i])
        agents[i].camera.draw_camera(ax, scale=0.2, color='brown')
    plt.savefig(name+'.pdf',bbox_inches='tight')
    plt.show()
    plt.close()
    
    #   TODO: Obtencion de error de rotación
    
    
    return t_err
    
    
def experiment(directory = "0",
               h  = 2.0,
               lamb = 1.0,
               depthOp=1,
               Z_set = 1.0,
               gdl = 1,
               t0 = 0.0,
               dt = 0.05,
               t_end = 10.0,
               zOffset = 0.0):
    
    #   Data
    P = np.array(ip.P)      #   Scene points
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
    
    p0 = np.array(ip.p0)    #   init positions
    p0[2,:] = p0[2,:]+zOffset
    n_agents = p0.shape[1] #Number of agents
    
    pd = ip.circle(n_agents,1.0,h)  #   Desired pose in a circle
    
    
    #   Parameters
    
    case_n = 1  #   Case selector if needed
    directed = False
    
    #Depth estimation for interaction matrix
    #   1-Updated, 2-Initial, 3-Referencia, 4-Arbitrary fixed, 5-Average
    depthOp_dict = {1:"Updated at each step",
                    2:"Distance at the begining",
                    3:"Distance between reference and points",
                    4:"Arbirtrary uniform value Z_set",
                    5:"Uniform value as the mean of all points"}
    
    #   interaction matrix used for the control
    #   1- computed each step 2- Computed at the end 3- average
    case_interactionM = 1
    case_interactionM_dict = {1:"Computed at each step",
                              2:"Computed at the end",
                              3:"Average between 1 and 2"}
    
    #   Controladores
    #   1  - IBVC   2 - Montijano
    control_type = 1
    control_type_dict = {1:"Image based visual control",
                         2:"Montijano"}
    case_controlable_dict = {1:"6 Degrees of freedom",
                             2:"4 Degrees of freedom",
                             3:"3 Degrees of freedom"}
    
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
    
    G = gr.graph(A_ad, directed)
    #G.plot()
    L = G.laplacian()
    
    #   Conectivity graph
    G.plot()
    
    
    [U,S,V]=svd(L)
    lam_n=S[0]
    lam_2=S[-2]
    alpha=2./(lam_n+lam_2)
    A_ds=np.eye(n_agents)-alpha*L
    
    #   Agents array
    agents = []
    for i in range(n_agents):
        cam = cm.camera()
        agents.append(ctr.agent(cam,pd[:,i],p0[:,i],P))
        
    #   INIT LOOP
    
    t=t0
    steps = int((t_end-t)/dt + 1.0)
    
    #   case selections
    if case_interactionM > 1:
        for i in range(n_agents):
            #   Depth calculation
            Z = Z_select(depthOp, agents[i], P,Z_set,p0,pd,j)
            if Z is None:
                return
            agents[i].set_interactionMat(Z,gdl)
    
    if control_type == 2:
        delta_pref = np.zeros((n_agents,n_agents,6,1))
        for i in range(n_agents):
            for j in range(n_agents):
                delta_pref[i,j,:,0] = pd[:,j]-pd[:,i]
        gamma =  p0[2,:]
    
    #   Storage variables
    t_array = np.arange(t,t_end+dt,dt)
    err_array = np.zeros((n_agents,2*n_points,steps))
    U_array = np.zeros((n_agents,6,steps))
    desc_arr = np.zeros((n_agents,2*n_points,steps))
    pos_arr = np.zeros((n_agents,3,steps))
    
    #   Print simulation data:
    print("------------------BEGIN-----------------")
    print("Laplacian selection = "+name)
    print("Is directed = "+str(directed))
    print(L)
    print(A_ds)
    print("Number of points = "+str(n_points))
    print("Number of agents = "+str(n_agents))
    if zOffset != 0.0:
        print("Z offset for starting conditions = "+str(zOffset))
    print("Time range = ["+str(t)+", "+str(dt)+", "+str(t_end)+"]")
    print("Control lambda = "+str(lamb))
    print("Depth estimation = "+depthOp_dict[depthOp])
    if depthOp == 4:
        print("\t Estimated theph set at : "+str(Z_set))
    print("Interaction matrix = "+case_interactionM_dict[case_interactionM])
    print("Control selection = "+control_type_dict[control_type])
    print("Controllable case = "+case_controlable_dict[gdl])
    
    #   LOOP
    for i in range(steps):
        
        print("loop",i, end="\r")
        
        #   Error:
        
        error = np.zeros((n_agents,2*n_points))
        for j in range(n_agents):
            error[j,:] = agents[j].error
        error = -L @ error
        
        #   save data
        #print(error)
        err_array[:,:,i] = error
        
        
        ####   Image based formation
        if control_type ==2:
            H = ctr.get_Homographies(agents)
        #   Get control
        for j in range(n_agents):
            
            #   save data 
            desc_arr[j,:,i] = agents[j].s_current.T.reshape(2*n_points)
            pos_arr[j,:,i] = agents[j].camera.p
            
            #   Depth calculation
            Z = Z_select(depthOp, agents[j], P,Z_set,p0,pd,j)
            if Z is None:
                return
            
            #   Control
            if control_type == 1:
                args = {"deg":G.deg[j] , 
                        "control_sel":case_interactionM,
                        "error": error[j,:],
                        "gdl":gdl}
            elif control_type == 2:
                args = {"H" : H[j,:,:,:],
                        "delta_pref" : delta_pref[j,:,:,:],
                        "Adj_list":G.list_adjacency[j][0],
                        "gamma": gamma[j]}
            else:
                print("invalid control selection")
                return
            
            U = agents[j].get_control(control_type,lamb,Z,args)
            
            if U is None:
                print("Invalid U control")
                break
            
            
            U_array[j,:,i] = U
            agents[j].update(U,dt,P)
            
        
        #   Update
        t += dt
        if control_type ==2:
            gamma = A_ds @ gamma #/ 10.0
        
    
    ##  Final data
    
    print("----------------------")
    print("Simulation final data")
    ret_err = np.zeros(n_agents)
    
    for j in range(n_agents):
        ret_err[j]=np.linalg.norm(error[j,:])
        print("|Error_"+str(j)+"|= "+str(ret_err[j]))
    for j in range(n_agents):
        print("X_"+str(j)+" = "+str(agents[j].camera.p))
    for j in range(n_agents):
        print("Angles_"+str(j)+" = "+str(agents[j].camera.roll)+
              ", "+str(agents[j].camera.pitch)+", "+str(agents[j].camera.yaw))
    
    
    ####   Plot
    # Colors setup
    
    #        RANDOM X_i
    colors = (randint(0,255,3*max(n_agents,2*n_points))/255.0).reshape((max(n_agents,2*n_points),3))
    
    new_agents = []
    for i in range(n_agents):
        cam = cm.camera()
        end_position = np.r_[agents[i].camera.p,agents[i].camera.roll, agents[i].camera.pitch, agents[i].camera.yaw]
        new_agents.append(ctr.agent(cam,pd[:,i],end_position,P))
    state_err = error_state(pd,new_agents,colors,directory+"/3D_error")
    print("State error = "+str(state_err))
    print("-------------------END------------------")
    print()
    
    
    
    #   Camera positions (init, end, ref) 
    lfact = 1.1
    mp.plot_position(pos_arr,
                    pd,
                    lfact,
                    colors,
                    name = directory+"/Cameras_trayectories")
                    #label = "Positions")
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    label = "World setting"
    name = directory+"/3Dplot"
    #fig.suptitle(label)
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    for i in range(n_agents):
        #agents[i].count_points_in_FOV(P)
        plot_3Dcam(ax, agents[i].camera,
                pos_arr[i,:,:],
                p0[:,i],
                pd[:,i],
                color = colors[i],
                camera_scale    = 0.02)
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   Descriptores (init, end, ref) x agente
    for i in range(n_agents):
        mp.plot_descriptors(desc_arr[i,:,:],
                            agents[i].camera.iMsize,
                            agents[i].s_ref,
                            colors,
                            name = directory+"/Image_Features_"+str(i),
                            label = "Image Features")
    
    #   Errores 
    for i in range(n_agents):
        mp.plot_time(t_array,
                    err_array[i,:,:],
                    colors,
                    name = directory+"/Features_Error_"+str(i),
                    label = "Features Error")
    
    #   Velocidaes x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    U_array[i,:,:],
                    colors,
                    name = directory+"/Velocidades_"+str(i),
                    label = "Velocidades",
                    labels = ["X","Y","Z","Wx","Wy","Wz"])
    return ret_err

def experiment_height():
    
    n_agents = 4
    
    #   Revisión con cambio uniforme de zOffset y h
    steps = 16
    var_arr = np.zeros((n_agents,steps))
    ref_arr = np.arange( 0.6, 0.6 + 0.2*(steps-0.9) ,0.2)
    
    for i in range(steps):
        ret_err = experiment(directory=str(i),depthOp = 1,
                   h = 0.6 + 0.2*i,
                   zOffset = -0.4 + 0.2*i,
                   t_end = 20)
        var_arr[:,i] = ret_err
    
    ##   Revisión con h=2.0
    #exp_select_z = [-0.4 -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    #var_arr = np.zeros((n_agents,len(exp_select_z)))
    #ref_arr = np.array(exp_select_z)
    #for i in range(len(exp_select_z)):
        #ret_err = experiment(directory=str(i),depthOp = 1,
                   #h = 2.0,
                   #zOffset = exp_select_z[i] ,
                   #t_end = 20)
        #var_arr[:,i] = ret_err
    
    ##   Revisión con h=1.0
    #exp_select_z = [-0.4, -0.2, 0., 0.2, 0.4, 0.6, 0.8, 1.0]
    #var_arr = np.zeros((n_agents,len(exp_select_z)))
    #ref_arr = np.array(exp_select_z)
    #for i in range(len(exp_select_z)):
        #ret_err = experiment(directory=str(i),depthOp = 1,
                   #h = 1.0,
                   #zOffset = exp_select_z[i] ,
                   #t_end = 20)
        #var_arr[:,i] = ret_err
        
    
    #   Plot data
    
    fig, ax = plt.subplots()
    fig.suptitle("Error de consenso")
    #plt.ylim([-2.,2.])
    
    colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    
    for i in range(n_agents):
        ax.plot(ref_arr,var_arr[i,:] , color=colors[i])
    
    plt.yscale('logit')
    plt.tight_layout()
    plt.savefig('Consensus error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
def main():
    
    ##   Experimentos de variación de altura
    #experiment_height()
    #return
    
    
    ##  Experimentos de variación de parámetros de contol
    #   Reference heights
    #exp_select = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    #   Lambda values
    #exp_select = [0.25, 0.5, 0.75, 1., 1.5, 1.25, 1.5, 1.75, 2., 5., 10., 15.]
    #   gdl
    #exp_select = [1, 2, 3]
    #   Z_set (depthOp = 4)
    exp_select = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    exp_select = [ 2.0 ]
    experiment(directory='0',depthOp = 1,
                    Z_set = 1.0,
                    h = 2.0,
                    lamb = 1,
                    gdl = 3,
                   zOffset = 1.0 ,
                   t_end = 10)
    
    return
    
    n_agents = 4
    var_arr = np.zeros((n_agents,len(exp_select)))
    ref_arr = np.array(exp_select)
    for i in range(len(exp_select)):
        ret_err = experiment(directory=str(i),depthOp = 1,
                    Z_set = exp_select[i],
                    lamb = 0.1,
                    gdl = 3,
                   #zOffset = 1.0 ,
                   t_end = 20)
        var_arr[:,i] = ret_err
        
    
    #   Plot data
    
    fig, ax = plt.subplots()
    fig.suptitle("Error de consenso")
    #plt.ylim([-2.,2.])
    
    colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    
    for i in range(n_agents):
        ax.plot(ref_arr,var_arr[i,:] , color=colors[i])
    
    plt.yscale('logit')
    plt.tight_layout()
    plt.savefig('Consensus error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()


if __name__ ==  "__main__":
    main()
