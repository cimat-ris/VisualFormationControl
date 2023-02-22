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



def plot_3Dcam(ax, camera,
               positionArray,
               init_configuration,
               desired_configuration,
               color,
               camera_scale    = 0.02):
    
    #ax.plot(positionArray[0,:],
    ax.scatter(positionArray[0,:],
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



def Z_select(depthOp, agent, P, Z_set, p0, pd, j):
    Z = np.ones((1,P.shape[1]))
    if depthOp ==1:
        #print(P)
        #   TODO creo que esto está mal calculado
        M = np.c_[ agent.camera.R.T, -agent.camera.R.T @ agent.camera.p ]
        Z = M @ P
        #print(Z)
        Z = Z[2,:]
        #if any(Z < 0.):
            #print(Z)
            #print(agent.camera.p )
            #print()
        #print(agent.camera.p )
        #print(agent.camera.roll )
        #print(agent.camera.pitch )
        #print(agent.camera.yaw )
        #print()
    elif depthOp == 6:
        Z = agent.camera.p[2]*np.ones(P.shape[1])
        Z = Z-P[2,:]
    elif depthOp ==2: # 
        Z = p0[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 3: # distancia al valor inicial
        Z = pd[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 4: # fijo al valor de Z_set
        Z = Z_set * np.ones(P.shape[1])
    elif depthOp == 5: # equitativo, = promedio
        tmp = agent.camera.p[2]-np.mean(P[2,:])
        Z = Z*tmp
    else:
        print("Invalid depthOp")
        return None
    return Z


    
    
def experiment(directory = "0",
               h  = 2.0,
               lamb = 1.0,
               depthOp=1,
               Z_set = 1.0,
               gdl = 1,
               t0 = 0.0,
               dt = 0.05,
               t_end = 10.0,
               zOffset = 0.0,
               case_interactionM = 1,
               p0 = np.array([0.,0.,1.,np.pi,0.,0.]),
               pd = np.array([0.,0.,1.,np.pi,0.,0.]),
               nameTag = "",
               tanhLimit = False):
    
    #   Data
    P = np.array(ip.P)      #   Scene points
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
    
    
    #   Parameters
    
    case_n = 1  #   Case selector if needed
    directed = False
    
    #Depth estimation for interaction matrix
    #   1-Updated, 2-Initial, 3-Referencia, 4-Arbitrary fixed, 5-Average
    depthOp_dict = {1:"Updated at each step",
                    2:"Distance at the begining",
                    3:"Distance between reference and points",
                    4:"Arbirtrary uniform value Z_set",
                    5:"Uniform value as the mean of all points",
                    6:"Height as depth"}
    
    #   interaction matrix used for the control
    #   1- computed each step 2- Computed at the end 3- average
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
    
    
    
    #   Agents array
    cam = cm.camera()
    agent = ctr.agent(cam,pd,p0,P)
        
    #   INIT LOOP
    
    t=t0
    steps = int((t_end-t)/dt + 1.0)
    
    #   case selections
    if case_interactionM > 1:
        #   Depth calculation
        Z = Z_select(depthOp, agent, P,Z_set,p0,pd,i)
        if Z is None:
            return
        agent.set_interactionMat(Z,gdl)
    
    #if control_type == 2:
        #delta_pref = np.zeros((n_agents,n_agents,6,1))
        #for i in range(n_agents):
            #for j in range(n_agents):
                #delta_pref[i,j,:,0] = pd[:,j]-pd[:,i]
        #gamma =  p0[2,:]
    
    #   Storage variables
    t_array = np.arange(t,t_end+dt,dt)
    steps = t_array.shape[0]
    err_array = np.zeros((1,2*n_points,steps))
    U_array = np.zeros((1,6,steps))
    desc_arr = np.zeros((1,2*n_points,steps))
    pos_arr = np.zeros((1,3,steps))
    if gdl == 1:
        s_store = np.zeros((1,6,steps))
    elif gdl == 2:
        s_store = np.zeros((1,4,steps))
    elif gdl == 3:
        s_store = np.zeros((1,3,steps))
    
    #   Print simulation data:
    print("------------------BEGIN-----------------")
    print("Reference state")
    print(pd)
    print("Initial state")
    print(p0)
    print("Scene points")
    print(P[:3,:])
    print("Number of points = "+str(n_points))
    print("Time range = ["+str(t)+", "+str(dt)+", "+str(t_end)+"]")
    print("Control lambda = "+str(lamb))
    print("Depth estimation = "+depthOp_dict[depthOp])
    if depthOp == 4:
        print("\t Estimated theph set at : "+str(Z_set))
    print("Interaction matrix = "+case_interactionM_dict[case_interactionM])
    print("Control selection = "+control_type_dict[control_type])
    print("Controllable case = "+case_controlable_dict[gdl])
    print("Total steps = "+str(steps))
    
    #   LOOP
    for i in range(steps):
        
        print("loop",i, end="\r")
        
        #   Error:
        error = agent.error.copy().reshape(2*n_points)
        
        #   save data
        #print(error)
        err_array[0,:,i] = error
        
        
        ####   Image based formation
        #if control_type ==2:
            #H = ctr.get_Homographies(agents)
        #   Get control
        
        
        #   save data 
        desc_arr[0,:,i] = agent.s_current.T.reshape(2*n_points)
        pos_arr[0,:,i] = agent.camera.p
        
        #   Depth calculation
        Z = Z_select(depthOp, agent, P,Z_set,p0,pd,0)
        if Z is None:
            return
        
        #   Control
        if control_type == 1:
            args = {"deg":1 , 
                    "control_sel":case_interactionM,
                    "error": error,
                    "gdl":gdl}
        #elif control_type == 2:
            #args = {"H" : H[j,:,:,:],
                    #"delta_pref" : delta_pref[j,:,:,:],
                    #"Adj_list":G.list_adjacency[j][0],
                    #"gamma": gamma[j]}
        else:
            print("invalid control selection")
            return
        
        #s = None
        U,u,s,vh  = agent.get_control(control_type,lamb,Z,args)
        s_store[0,:,i] = s
        if tanhLimit:
            U = 0.3*np.tanh(U)
            #U[:3] = 0.5*np.tanh(U[:3])
            #U[3:] = 0.3*np.tanh(U[3:])
        #print(s)
        #U[abs(U) > 0.2] = np.sign(U)[abs(U) > 0.2]*0.2
        #U_sel = abs(U[3:]) > 0.2
        #U[3:][U_sel] = np.sign(U[3:])[U_sel]*0.2
        
        if U is None:
            print("Invalid U control")
            break
        
        
        U_array[0,:,i] = U
        agent.update(U,dt,P, Z)
        
    
        #   Update
        t += dt
        #if control_type ==2:
            #gamma = A_ds @ gamma #/ 10.0
        
    
    ##  Final data
    
    print("----------------------")
    print("Simulation final data")
    
    ret_err=np.linalg.norm(error)
    print(error)
    print("|Error|= "+str(ret_err))
    print("X = "+str(agent.camera.p))
    print("Angles = "+str(agent.camera.roll)+
            ", "+str(agent.camera.pitch)+", "+str(agent.camera.yaw))
    
    
    ####   Plot
    # Colors setup
    
    #        RANDOM X_i
    colors = (randint(0,255,3*2*n_points)/255.0).reshape((2*n_points,3))
    
    #cam = cm.camera()
    #end_position = np.r_[agent.camera.p,agent.camera.roll, agent.camera.pitch, agent.camera.yaw]
    #new_agent = ctr.agent(cam,pd,end_position,P)
    #state_err = error_state(pd,new_agent,colors,directory+"/3D_error")
    #print("State error = "+str(state_err))
    print("-------------------END------------------")
    print()
    
    
    pd = pd.reshape((6,1))
    p0 = p0.reshape((6,1))
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
    #agents[i].count_points_in_FOV(P)
    plot_3Dcam(ax, agent.camera,
            pos_arr[0,:,:],
            p0[:,0],
            pd[:,0],
            color = colors[0],
            camera_scale    = 0.02)
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   Descriptores (init, end, ref) x agente
    print("Agent: "+nameTag)
    L = ctr.Interaction_Matrix(agent.s_current_n,Z,gdl)
    A = L.T@L
    if np.linalg.det(A) != 0:
        print("Matriz de interacción (L): ")
        print(L)
        L = inv(A) @ L.T
        print("Matriz de interacción pseudoinversa (L+): ")
        print(L)
        print("Error de consenso final (e): ")
        print(error)
        print("velocidades resultantes (L+ @ e): ")
        print(L@error)
        print("Valores singulares al final (s = SVD(L+)): ")
        print(s_store[0,:,-1])
    mp.plot_descriptors(desc_arr[0,:,:],
                        agent.camera.iMsize,
                        agent.s_ref,
                        colors,
                        name = directory+"/Image_Features_"+nameTag,
                        label = "Image Features")
    
    #   Errores 
    mp.plot_time(t_array,
                err_array[0,:,:],
                colors,
                name = directory+"/Features_Error_"+nameTag,
                label = "Features Error")
    
    #   Velocidaes x agente
    mp.plot_time(t_array,
                U_array[0,:,:],
                colors,
                name = directory+"/Velocidades_"+nameTag,
                label = "Velocidades",
                labels = ["X","Y","Z","Wx","Wy","Wz"])
    
    #   Valores propios
    mp.plot_time(t_array,
                s_store[0,:,:],
                colors,
                name = directory+"/ValoresP_"+nameTag,
                label = "Valores propios (SVD)",
                labels = ["0","1","2","3","4","5"])
                #limits = [[t_array[0],t_array[-1]],[0,20]])
    return ret_err, pos_arr[0,:,:], agent.camera

def experiment_height():
    
    n_agents = 4
    
    #   Revisión con cambio uniforme de zOffset y h
    #   Casos 1
    #ref_arr = np.arange( 1., 2.55 ,0.1)
    #ref_arr = np.arange( 1., 2.25 ,0.2)
    #   Casos 2
    #ref_arr = np.arange( 0.5, 2.05 ,0.1)
    #   Caso 3
    #ref_arr = np.arange( 0.6, 2.15 ,0.1)
    
    
    #var_arr = np.zeros((n_agents,len(ref_arr)))
    #var_arr_2 = np.zeros(len(ref_arr))
    #var_arr_3 = np.zeros(len(ref_arr))
    
    #for i in range(len(ref_arr)):
        #ret_err = experiment(directory=str(i),
                             #gdl = 1,
                             #h = ref_arr[i],
                             #zOffset = ref_arr[i] -0.8 ,
                             #t_end = 60,
                             #tanhLimit = True)
        #[var_arr[:,i], var_arr_2[i], var_arr_3[i]] = ret_err
    
   
    
    #   Revisión con zOffset constante
    #   Offset = 0
    #offset = 0.
    ##       Caso 1
    #ref_arr = np.arange( 1.0, 2.05 ,0.1)
    #ref_arr = np.arange( 0.6, 1.55 ,0.2)
    ##       Caso 2 = puntos coplanares
    #ref_arr = np.arange( 0.4, 1.75 ,0.1)
    ##       Caso 3 = 4 dof
    #ref_arr = np.arange( 0.6, 1.75 ,0.1)
    #ref_arr = np.arange( 1.1, 1.75 ,0.2)
    
    #   Offset = 1
    #offset = 1.
    
    #       Caso 1
    #ref_arr = np.arange( 1.3, 2.0 ,0.1)
    #ref_arr = np.arange( 1.0, 1.5 ,0.1)
    
    #       Caso 2, 3
    #ref_arr = np.arange( 0.7, 2.0 ,0.1)
    
    #   h = 1.
    
    #   Caso 1
    #ref_arr = np.arange( 0.6, 2.05 ,0.1)
    ref_arr = np.arange( 0.6, 1.75 ,0.1)
    #ref_arr = np.arange( 0.6, 1.15 ,0.1)
    
    
    var_arr = np.zeros((n_agents,len(ref_arr)))
    var_arr_2 = np.zeros(len(ref_arr))
    var_arr_3 = np.zeros(len(ref_arr))
    #ref_arr = np.array(ref_arr)
    for i in range(len(ref_arr)):
        ret_err = experiment(directory=str(i),
                             lamb = .2,
                             gdl = 1,
                             #zOffset = offset,
                             zOffset = ref_arr[i],
                             #h = ref_arr[i] ,
                             h = 1. ,
                             #tanhLimit = True,
                             #depthOp = 4, Z_set=1.,
                             t_end = 80)
                             #t_end = 20)
        [var_arr[:,i], var_arr_2[i], var_arr_3[i]] = ret_err
        
    
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
    
    fig, ax = plt.subplots()
    fig.suptitle("Errores de estado")
    ax.plot(ref_arr,var_arr_2, label = "Posición")
    ax.plot(ref_arr,var_arr_3, label = "Rotación")
    fig.legend( loc=2)
    plt.tight_layout()
    plt.savefig('Formation error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    print("Total number of simulations = "+str(len(ref_arr)))

def experiment_localmin():
    
    t = 100
    #t = 20
    
    ##   Casos con formación 
    ##   Caso minimo local e == 0 
    ##   P_z = [0.0,  -0.2, 0.5]
    #experiment(directory='0',
                    #h = 1.5,
                    #lamb = 1.,
                    #gdl = 1,
                   #zOffset = 0.7 ,
                   #t_end = t)
    
    ###   Caso minimo local e != 0 
    ###   P_z = [0.0,  -0.2, 0.5]
    #experiment(directory='1',
                    #h = 1.3,
                    #lamb = 1.,
                    #gdl = 1,
                   #zOffset = 1.0 ,
                   #t_end = t)
    
    #   Casos solo consenso
    #   Caso minimo local e == 0 
    #   P_z = [0.0,  -0.2, 0.5]
    experiment(directory='0',
                    h = 1.5,
                    lamb = 1.,
                    gdl = 1,
                   zOffset = 0.7 ,
                   t_end = t)
    
    #   Caso minimo local e != 0 
    #   P_z = [0.0,  -0.2, 0.5]
    experiment(directory='1',
                    h = 0.6,
                    lamb = 1.,
                    gdl = 1,
                   zOffset = -.2 ,
                   t_end = t)

def experiment_mesh(x,y,z):
    _x, _y = np.meshgrid(x,y)
    n = _x.size
    
    pd = np.array([0.,0.,1.,np.pi,0.,0.])
    p0 = np.array([_x.reshape(n),
                     _y.reshape(n),
                     z*np.ones(n),
                     np.pi*np.ones(n),
                     np.zeros(n),
                     np.zeros(n)])
    #n = mesh.shape[1]
    print("testing ",n," repeats")
    #i = 0
    #p0 = np.r_[mesh,z*np.ones(n),np.pi*np.ones(n),np.zeros(n),np.zeros(n)]
    pos_arr = []
    cam_arr = []
    for i in range( n):
        err, _pos_arr, _cam = experiment(directory=str(i),
                                lamb = 1.,
                                gdl = 1,
                                #zOffset = 0.6,
                                h = 1. ,
                                pd = pd,
                                p0 = p0[:,i],
                                #tanhLimit = True,
                                #depthOp = 4, Z_set=1.,
                                #depthOp = 6,
                                t_end = 10.)
        pos_arr.append(_pos_arr)
        cam_arr.append(_cam)
    
    ##  Prepare data for plot
    colors = (randint(0,255,3*n)/255.0).reshape((n,3))
    P = np.array(ip.P)      #   Scene points
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
    
    #   plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = "3Dplot"
    #fig.suptitle(label)
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    #agents[i].count_points_in_FOV(P)
    for i in range( n):
        plot_3Dcam(ax, cam_arr[i],
                pos_arr[i],
                p0[:,i],
                pd,
                color = colors[0],
                camera_scale    = 0.02)
    plt.savefig(name+'.pdf',bbox_inches='tight')
    plt.show()
    plt.close()
    
def main():
    
    x = np.linspace(-1,1,2)
    y = np.linspace(-1,1,2)
    experiment_mesh(x,y,z=2)
    return
    
    p0=np.array([1.,1.,2.,np.pi,0.,0.])
    p0=np.array([1.,1.,2.,np.pi+0.5,0.5,1.])
    experiment(directory='0',
                lamb = 1.,
                gdl = 1,
                #zOffset = 0.6,
                h = 1. ,
                p0 = p0,
                #tanhLimit = True,
                #depthOp = 4, Z_set=1.,
                #depthOp = 6,
                t_end = 10.)
    return
    
    #   Caso minimo local e != 0 
    #   P_z = [0.0,  -0.2, 0.5]
    #experiment(directory='14',
                    #h = 1.1,
                    #lamb = 0.1,
                    #gdl = 1,
                   #zOffset = 0.0 ,
                   #t_end = 100,
                   #tanhLimit = True)
    #experiment(directory='12',
                    #gdl = 1,
                    #h = 1.5,
                    #zOffset = 0.7 ,
                    #t_end = 90,
                    #tanhLimit = True)
    #return
    
    ##   Experimentos de variación de altura
    experiment_height()
    #return
    
    ##  Experimentos de minimos locales
    #experiment_localmin()
    return
    
    
    ##  Experimentos de variación de parámetros de contol
   
    #   Lambda values
    #exp_select = [0.25, 0.5, 0.75, 1., 1.5, 1.25, 1.5, 1.75, 2., 5., 10., 15.]
    #   gdl
    #exp_select = [1, 2, 3]
    #   Z_set (depthOp = 4)
    exp_select = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    exp_select = [ 2.0 ]
    
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
