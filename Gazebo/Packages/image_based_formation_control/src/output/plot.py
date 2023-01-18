"""
    Python script to plot all the given information from the simulator
    email: patriciatt.tavares@cimat.mx
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import os

#   Simulation data
n = 3 #number of quadrotors
reference = np.array([[1,3,2],
                      [3,1,2],
                      [1,1,2]],dtype = float)

#   Parameters
init = 1
error_case = 0 
# 0 = constant error dimention
# 1 = non constant error dimention

#plot the information from every folder
for i in range(n):
    print('Processing '+str(i)+'...')
    dir = str(i)
    for j in range(n):
        
        
        data = np.loadtxt(dir+'/partial_'+str(j)+'.txt')
        time = data[init:,0]
        
        if (i == j ):
            pose = data[init:,[1,2,3,4]]
            
            #plot translation pose
            plt.plot(time, pose[:,0],label = "X")
            plt.plot(time, pose[:,1],label = "Y")
            plt.plot(time, pose[:,2],label = "Z")
            plt.ylabel('Pose $(m)$')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            #plt.ylim((-1,1))
            plt.legend(loc=0)
            plt.savefig(dir+'/state_t_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()
            
            #plot rotation pose
            plt.plot(time, pose[:,3],label = "Yaw")
            plt.ylabel('Rotation $(rad)$')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            plt.ylim((-1,1))
            plt.legend(loc=0)
            plt.savefig(dir+'/state_yaw_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()
            
            #plot translation error
            pose[:,0] = pose[:,0] - reference[i,0]
            pose[:,1] = pose[:,1] - reference[i,1]
            pose[:,2] = pose[:,2] - reference[i,2]
            plt.plot(time, pose[:,0],label = "X")
            plt.plot(time, pose[:,1],label = "Y")
            plt.plot(time, pose[:,2],label = "Z")
            plt.ylabel('Error $(m)$')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            plt.ylim((-1,1))
            plt.legend(loc=0)
            plt.savefig(dir+'/error_t_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()
            
            #   step Time histogram
            dt = time[1:]-time[:-1]
            plt.hist(dt,bins=int(dt.max()/0.05))
            plt.savefig(dir+'/dt_histogram.pdf',bbox_inches='tight')
            plt.clf()
        
        velocities = data[init:,[7,8,9,10]]
    
        #plot translation velocities
        plt.plot(time, velocities[:,0],label = "V_x")
        plt.plot(time, velocities[:,1],label = "V_y")
        plt.plot(time, velocities[:,2],label = "V_z")
        plt.ylabel('Velocity $(m/s)$')
        plt.xlabel('Time $(s)$')
        plt.grid(True)
        plt.ylim((-1,1))
        plt.legend(loc=0)
        plt.savefig(dir+'/velocity_t_'+str(i)+"_"+str(j)+'.pdf',bbox_inches='tight')
        plt.clf()
        
        #plot rotation velocities
        plt.plot(time, velocities[:,3],label = "Yaw")
        plt.ylabel('Rotation velocoty $(rad/s)$')
        plt.xlabel('Time $(s)$')
        plt.grid(True)
        plt.ylim((-1,1))
        plt.legend(loc=0)
        plt.savefig(dir+'/velocity_yaw_'+str(i)+"_"+str(j)+'.pdf',bbox_inches='tight')
        plt.clf()
        
        #   Plot consensus error
        
        if error_case == 0:
            
            #   case constant descriptor dimention
            data = np.loadtxt(dir+'/error_'+str(j)+'.txt')
            time = data[init:,0]
            dimention = data.shape[1]
            
            #plot error u
            error = data[init:,range(1,dimention,2)]/241.4268236
            #error = data[init:,range(1,int((1+dimention)/2))]
            plt.plot(time, error)
            plt.ylabel('U error $(m)$')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            plt.ylim((-1,1))
            plt.savefig(dir+'/error_u_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()
            
            #plot error v
            error = data[init:,range(2,dimention,2)]/241.4268236
            #error = data[init:,range(int((1+dimention)/2),dimention)]
            plt.plot(time, error)
            plt.ylabel('V error $(m)$')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            plt.ylim((-1,1))
            plt.savefig(dir+'/error_v_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()
            
            
        elif i != j:
            #   case non constant descriptor dimention
            
            file1 = open(dir+'/error_'+str(j)+'.txt', 'r')
            Lines = file1.readlines()
                
            # Strips the newline character
            _average = np.zeros(time.shape)
            _maximum = np.zeros(time.shape)
            _minimum = np.zeros(time.shape)
            _counter = np.zeros(time.shape)
            counter = 0
            for line in Lines:
                l = line.strip()
                l = l.split()
                d = np.array(l)
                d = d.astype(np.float64)
                _average[counter] = d.mean()
                _minimum[counter] = d.min()
                _maximum[counter] = d.max()
                _counter[counter] = d.shape[0]
                
                counter += 1

            #plot consensus error
            plt.plot(time,_average, label = "average")
            plt.plot(time,_minimum, label = "min")
            plt.plot(time,_maximum, label = "max")
            plt.ylabel('Consensus error')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            plt.ylim((-1,1))
            plt.legend(loc=0)
            plt.savefig(dir+'/error_consensus_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()
            
            plt.plot(time,_counter/2.0, label = "points")
            plt.ylabel('Point counter')
            plt.xlabel('Time $(s)$')
            plt.grid(True)
            plt.ylim((-1,1))
            plt.legend(loc=0)
            plt.savefig(dir+'/Point_counter_'+str(j)+'.pdf',bbox_inches='tight')
            plt.clf()

    
        
    print ('Done.')
        
