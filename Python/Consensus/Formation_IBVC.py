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
               color, i,
               camera_scale    = 0.02):
    
    #ax.plot(positionArray[0,:],
    ax.scatter(positionArray[0,:],
            positionArray[1,:],
            positionArray[2,:],
            label = str(i),
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


#   Error if state calculation
#   It assumes that the states in the array are ordered and are the same
#   regresa un error de traslación y orientación
def error_state(reference,  agents, name):
    
    n = reference.shape[1]
    state = np.zeros((6,n))
    for i in range(len(agents)):
        state[:3,i] = agents[i].camera.p
        state[3,i] = agents[i].camera.roll
        state[4,i] = agents[i].camera.pitch
        state[5,i] = agents[i].camera.yaw
    
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
    #theta  = 0.
    
    ca = cos(theta)
    sa = sin(theta)
    R = np.array([[ ca, -sa],
                  [ sa,  ca]])
    new_state[:2,:] = R@new_state[:2,:]
    
    #       Updtate normalized and oriented agents
    for i in range(len(agents)):
        #new_p = np.r_[new_state[:,i] ,agents[i].camera.roll,agents[i].camera.pitch,agents[i].camera.yaw + theta]
        #agents[i].camera.pose(new_p)
        new_state[5,i] += theta
        agents[i].camera.pose(new_state[:,i])
        
    #   normalizar referencia
    new_reference[:3,:] /= np.linalg.norm(new_reference[:3,:],axis=0).mean()
    
    #   Aplicar minimización de radio (como lidiar con mínimos múltiples)
    f = lambda r : (np.linalg.norm(new_reference[:3,:] - r*new_state[:3,:],axis = 0)**2).sum()/n
    r_state = minimize_scalar(f, method='brent')
    t_err = f(r_state.x)
    
    #   Plot
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    new_state[:3,:] *= r_state.x
    for i in range(n):
        agents[i].camera.pose(new_state[:,i])
        agents[i].camera.draw_camera(ax, scale=0.2, color='red')
        agents[i].camera.pose(new_reference[:,i])
        agents[i].camera.draw_camera(ax, scale=0.2, color='brown')
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   TODO: Obtencion de error de rotación
    rot_err = np.zeros(n)
    for i in range(n):
        _R =  cm.rot(new_state[3,i],'x') @ agents[i].camera.R.T
        _R = cm.rot(new_state[4,i],'y') @ _R
        _R = cm.rot(new_state[5,i],'z') @ _R
        rot_err = np.arccos((_R.trace()-1.)/2.)
    rot_err = rot_err**2
    rot_err = rot_err.sum()/n
    
    return [t_err, rot_err]

#   Difference between agents
#   It assumes that the states in the array are ordered and are the same
#   regresa un error de traslación y orientación
def error_state_equal(  agents, name):
    
    n = len(agents)
    state = np.zeros((6,n))
    for i in range(n):
        state[:3,i] = agents[i].camera.p
        state[3,i] = agents[i].camera.roll
        state[4,i] = agents[i].camera.pitch
        state[5,i] = agents[i].camera.yaw
    
    reference = np.average(state,axis =1)
    
    t_err =  (np.linalg.norm(reference[:3].reshape((3,1)) - state[:3,:],axis = 0)**2).sum()/n
    rot_err =  (np.linalg.norm(reference[3:].reshape((3,1)) - state[3:,:],axis = 0)**2).sum()/n
    
    #   Plot
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(n):
        agents[i].camera.draw_camera(ax, scale=0.2, color='red')
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    return [t_err, rot_err]
    
    
def experiment(directory = "0",
               h  = 2.0,
               r = 1.0,
               lamb = 1.0,
               k_int = 0,
               depthOp=1,
               Z_set = 1.0,
               gdl = 1,
               t0 = 0.0,
               dt = 0.05,
               t_end = 10.0,
               zOffset = 0.0,
               case_interactionM = 1,
               p0 = None,
               pd = None,
               set_consensoRef = True,
               atTarget = False,
               tanhLimit = False):
    
    #   Data
    P = np.array(ip.P)      #   Scene points
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
    
    #   Posiciones de inicio
    if p0 is None:
        p0 = np.array(ip.p0)    
        p0[2,:] = p0[2,:]+zOffset
    n_agents = p0.shape[1] #Number of agents
    
    if pd is None:
        pd = ip.circle(n_agents,r,h)  #   Desired pose in a circle
    
    if atTarget:
        p0[:3,:] = pd[:3,:]
    
    if pd.shape != p0.shape:
        print("Error: Init and reference position missmatch")
        return 
    
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
        agents.append(ctr.agent(cam,pd[:,i],p0[:,i],P,
                                k_int = k_int,
                                set_consensoRef = set_consensoRef))
        
    #   INIT LOOP
    
    t=t0
    steps = int((t_end-t)/dt + 1.0)
    
    #   case selections
    if case_interactionM > 1:
        for i in range(n_agents):
            #   Depth calculation
            Z = Z_select(depthOp, agents[i], P,Z_set,p0,pd,i)
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
    pos_arr = np.zeros((n_agents,6,steps))
    if gdl == 1:
        s_store = np.zeros((n_agents,6,steps))
    elif gdl == 2:
        s_store = np.zeros((n_agents,4,steps))
    elif gdl == 3:
        s_store = np.zeros((n_agents,3,steps))
    
    #   Print simulation data:
    print("------------------BEGIN-----------------")
    print("Laplacian selection = "+name)
    print("Is directed = "+str(directed))
    print("Laplacian matrix: ")
    print(L)
    print(A_ds)
    print("Reference states")
    print(pd)
    print("Initial states")
    print(p0)
    print("Scene points")
    print(P[:3,:])
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
        error_p = np.zeros((n_agents,2*n_points))
        for j in range(n_agents):
            error[j,:] = agents[j].error
            error_p[j,:] = agents[j].error_p
        error = L @ error
        error_p = L @ error_p
        
        #   save data
        #print(error)
        err_array[:,:,i] = error_p
        
        
        ####   Image based formation
        if control_type ==2:
            H = ctr.get_Homographies(agents)
        #   Get control
        for j in range(n_agents):
            
            #   save data 
            desc_arr[j,:,i] = agents[j].s_current.T.reshape(2*n_points)
            pos_arr[j,:,i] = np.r_[agents[j].camera.p.T ,
                                   agents[j].camera.roll,
                                   agents[j].camera.pitch,
                                   agents[j].camera.yaw]
            
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
            
            #s = None
            U,u, s, vh  = agents[j].get_control(control_type,lamb,Z,args)
            s_store[j,:,i] = s
            if tanhLimit:
                U = 0.3*np.tanh(U)
            
            #if any(abs(error[j,:])> 1.):
                #print("----ERR>--")
                #print("i ",i)
                #print("j ",j)
                #print("error ",error[j,:])
                #print("U ",U)
            #if any(abs(U) > 10.):
                #print("---UPS---")
                #print("i ",i)
                #print("j ",j)
                #print("U ",U)
            if(s[0] > 200):
                #print("PVAL > 1000")
                print("-----JUMP------")
                print("i ",i)
                print("j ",j)
                print("Z  ",Z)
                print("L ",ctr.Interaction_Matrix(agents[j].s_current_n,Z,gdl))
                print("Valores propio ",s)
                print("Puntos normalizados ",agents[j].s_current_n)
                print("Puntos en pixeles ",agents[j].s_current)
                print("Posicion  ",agents[j].camera.p)
                print("error  ",error[j,:])
                print("Vector propio 0  ",u[:,0])
                #print("Prod int u,err ",u.T@error[j,:]/np.linalg.norm(error[j,:]))
                #print("Prod int v,err ",vh@error[j,:]/np.linalg.norm(error[j,:]))
                print("U ",U)
                print("---------------")
            
            if U is None:
                print("Invalid U control")
                break
            
            
            U_array[j,:,i] = U
            agents[j].update(U,dt,P, Z)
            
        
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
        print(error[j,:])
        print("|Error_"+str(j)+"|= "+str(ret_err[j]))
    for j in range(n_agents):
        print("X_"+str(j)+" = "+str(agents[j].camera.p))
    for j in range(n_agents):
        print("V_"+str(j)+" = "+str(U_array[j,:,-1]))
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
    if set_consensoRef:
        state_err = error_state(pd,new_agents,directory+"/3D_error")
    else:
        state_err = error_state_equal(new_agents,directory+"/3D_error")
        
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
                i = i,
                camera_scale    = 0.02)
    
    #f_size = 1.1*max(abs(p0[:2,:]).max(),abs(pd[:2,:]).max())
    lfact = 1.1
    x_min = lfact*min(p0[0,:].min(),pd[0,:].min())
    x_max = lfact*max(p0[0,:].max(),pd[0,:].max())
    y_min = lfact*min(p0[1,:].min(),pd[1,:].min())
    y_max = lfact*max(p0[1,:].max(),pd[1,:].max())
    z_max = lfact*max(p0[2,:].max(),pd[2,:].max())
    
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(0,z_max)
    
    fig.legend( loc=1)
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   Descriptores (init, end, ref) x agente
    for i in range(n_agents):
        print("Agent: "+str(i))
        L = ctr.Interaction_Matrix(agents[i].s_current_n,Z,gdl)
        A = L.T@L
        if np.linalg.det(A) != 0 and ret_err[i] > 1.e-2:
            print("Matriz de interacción (L): ")
            print(L)
            L = inv(A) @ L.T
            print("Matriz de interacción pseudoinversa (L+): ")
            print(L)
            print("Error de consenso final (e): ")
            print(error[i,:])
            print("velocidades resultantes (L+ @ e): ")
            print(L@error[i,:])
            print("Valores singulares al final (s = SVD(L+)): ")
            print(s_store[i,:,-1])
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
    
    #   Posiciones x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    pos_arr[i,:3,:],
                    colors,
                    name = directory+"/Traslaciones_"+str(i),
                    label = "Traslaciones",
                    labels = ["X","Y","Z"])
        mp.plot_time(t_array,
                    pos_arr[i,3:,:],
                    colors,
                    name = directory+"/Angulos_"+str(i),
                    label = "Angulos",
                    labels = ["Roll","Pitch","yaw"])
    
    #   Valores propios
    for i in range(n_agents):
        mp.plot_time(t_array,
                    s_store[i,:,:],
                    colors,
                    name = directory+"/ValoresP_"+str(i),
                    label = "Valores propios (SVD)",
                    labels = ["0","1","2","3","4","5"])
                    #limits = [[t_array[0],t_array[-1]],[0,20]])
    return [ret_err, state_err[0],state_err[1]]

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

def experiment_randomInit(justPlot = False,
                          r = 0.8,
                        n = 1,
                        k_int = 0.1,
                        t_f = 100):
    
    if justPlot:
        
        #   Load
        arr_error = np.load('arr_err.npy')
        arr_epsilon = np.load('arr_epsilon.npy')
        n = arr_epsilon.shape[2]
        ref_arr = np.arange(n)
        
    else:
        
        r = 0.8
        n = 1
        k_int = 0.1
        t_f = 100
        
        ref_arr = np.arange(n)
        #   4 variantes X 4 agentes X n repeticiones
        arr_error = np.zeros((4,4,n))
        #   4 variantes X 2 componentes X n repeticiones
        arr_epsilon = np.zeros((4,2,n))
        for i in range(n):
            #p0=[[0.8,0.8,-0.8,-.8],
                #[-0.8,0.8,0.8,-0.8],
            p0=[[-0.8,0.8,0.8,-.8],
                [0.8,0.8,-0.8,-0.8],
                #[1.4,0.8,1.2,1.6],
                [1.2,1.2,1.2,1.2],
                [np.pi,np.pi,np.pi,np.pi],
                [0,0,0,0],
                [0,0,0,0]]
            
            p0 = np.array(p0)
            p0[:3,:] += r *2*( np.random.rand(3,p0.shape[1])-0.5)
            
            ret = experiment(directory=str(i*4),
                        k_int =k_int,
                        h = 1 ,
                        r = 1.,
                        #tanhLimit = True,
                        #depthOp = 4, Z_set=2.,
                        depthOp = 1,
                        p0 = p0,
                        t_end = t_f)
            
            [arr_error[0,:,i], arr_epsilon[0,0,i], arr_epsilon[0,1,i]] = ret
            
            ret = experiment(directory=str(i*4+1),
                        k_int = k_int,
                        h = 1 ,
                        r = 1.,
                        #tanhLimit = True,
                        #depthOp = 4, Z_set=2.,
                        depthOp = 1,
                        p0 = p0,
                        set_consensoRef = False,
                        t_end = 20)
            [arr_error[1,:,i], arr_epsilon[1,0,i], arr_epsilon[1,1,i]] = ret
            ret = experiment(directory=str(i*4+2),
                        h = 1 ,
                        r = 1.,
                        #tanhLimit = True,
                        #depthOp = 4, Z_set=2.,
                        depthOp = 1,
                        p0 = p0,
                        t_end = t_f)
            
            [arr_error[2,:,i], arr_epsilon[2,0,i], arr_epsilon[2,1,i]] = ret
            
            ret = experiment(directory=str(i*4+3),
                        h = 1 ,
                        r = 1.,
                        #tanhLimit = True,
                        #depthOp = 4, Z_set=2.,
                        depthOp = 1,
                        p0 = p0,
                        set_consensoRef = False,
                        t_end = 20)
            [arr_error[3,:,i], arr_epsilon[3,0,i], arr_epsilon[3,1,i]] = ret
        np.save('arr_err.npy',arr_error)
        np.save('arr_epsilon.npy',arr_epsilon)
        
        
    #   Plot data
    
    fig, ax = plt.subplots()
    fig.suptitle("Error de consenso")
    #plt.ylim([-2.,2.])
    
    colors = (randint(0,255,3*4*4)/255.0).reshape((4*4,3))
    for i in range(4):
        ax.scatter(ref_arr,arr_error[0,i,:], marker = "x", alpha = 0.5, color=colors[0])
        ax.scatter(ref_arr,arr_error[1,i,:], marker = "x", alpha = 0.5, color=colors[1])
        ax.scatter(ref_arr,arr_error[2,i,:], marker = "x", alpha = 0.5, color=colors[2])
        ax.scatter(ref_arr,arr_error[3,i,:], marker = "x", alpha = 0.5, color=colors[3])
    
    symbols = [mpatches.Patch(color=colors[0]),
               mpatches.Patch(color=colors[1]),
               mpatches.Patch(color=colors[2]),
               mpatches.Patch(color=colors[3])]
    fig.legend(symbols,["Ref (PI)","No ref (PI)","Ref (P)","No ref (P)"], loc=1)
    plt.yscale('logit')
    plt.tight_layout()
    plt.savefig('Consensus error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    fig, ax = plt.subplots()
    fig.suptitle("Errores de estado")
    ax.scatter(ref_arr,arr_epsilon[0,0,:], marker='.', label  = "Ref (PI) Tras", alpha = 0.5, color = colors[0])
    ax.scatter(ref_arr,arr_epsilon[0,1,:], marker='*', label  = "Ref (PI) Rot", alpha = 0.5, color = colors[0])
    ax.scatter(ref_arr,arr_epsilon[1,0,:], marker='.', label  = "No ref (PI) Tras", alpha = 0.5, color = colors[1])
    ax.scatter(ref_arr,arr_epsilon[1,1,:], marker='*', label  = "No ref (PI) Rot", alpha = 0.5, color = colors[1])
    ax.scatter(ref_arr,arr_epsilon[2,0,:], marker='.', label  = "Ref (P) Tras", alpha = 0.5, color = colors[2])
    ax.scatter(ref_arr,arr_epsilon[2,1,:], marker='*', label  = "Ref (P) Tras Rot", alpha = 0.5, color = colors[2])
    ax.scatter(ref_arr,arr_epsilon[3,0,:], marker='.', label  = "No ref (P) Tras", alpha = 0.5, color = colors[3])
    ax.scatter(ref_arr,arr_epsilon[3,1,:], marker='*', label  = "No ref (P) Rot", alpha = 0.5, color = colors[3])
    
    plt.ylim([0,0.01])
    fig.legend( loc=1)
    plt.tight_layout()
    plt.savefig('Formation error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
def main():
    
    #experiment_randomInit(n = 10)
    experiment_randomInit(justPlot = True)
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
