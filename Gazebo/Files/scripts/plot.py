import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
# import re
import os

from numpy import pi

#   Configs

get_type={2:np.float16,
          4:np.float32,
          8:np.float64}

def plot_time(ax, t_array,
              var_array,
              ref = None,
              module = None):

    n = var_array.shape[0]

    npzfile = np.load("general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]



    symbols = []
    for i in range(n):
        ax.plot(t_array,var_array[i,:] , color=colors[i%nColors])
        symbols.append(mpatches.Patch(color=colors[i%nColors]))

    if not ref is None:
        ax.plot([t_array[0],t_array[-1]],[ref,ref],
                'k--', alpha = 0.5)
        symbols.append(mpatches.Patch(color='k'))
        labels.append(refLab)
    if not module is None:
        for i in range(len(module)):
            ax.plot([t_array[0],t_array[-1]],[module[i],module[i]],
                'r--', lw = 0.5)
        symbols.append(mpatches.Patch(color='r'))
        labels.append("Limits")

    return symbols

def main(arg):
    n_agents = 4
    pd=np.array([[0.,0.,0.,0.],
                [0.,0.,0.,0.],
                [0.,0.,0.,0.],
                [pi,pi,pi,pi],
                [0.,0.,0.,0.],
                [0.,0.,0.,0.]])


    for i in range(n_agents):

        #   read errors data
        directory = arg[1].rstrip('/')+'/'
        name = directory + str(i)+'/error.dat'
        length = os.path.getsize(name)
        print("length = ", length)
        ArUcoTab = {}

        with open(name, 'rb') as fileH:
            #   header
            row_bytes = np.fromfile(fileH, dtype=np.int32, count= 1)
            # row_bytes = row_bytes[0]
            print("row_bytes = ", row_bytes)
            # elemSize = (row_bytes-4-8)/4
            # print(elemSize)
            # elemSize = int(elemSize)
            # print(elemSize)
            # rows = (length-4) / row_bytes
            rows = (length-4) / (8*10)
            # print(rows)
            rows = int(np.floor(rows))
            print("rows = ",rows)
            # Read the data into a NumPy array

            #   TODO np.dtype method
            for j in range(rows):
                timestamp = np.fromfile(fileH,
                                    dtype=np.float64,
                                    count = 1)
                ArUcoId = np.fromfile(fileH,
                                    dtype=np.int32,
                                    count = 1)
                ArUcoId = ArUcoId[0]
                errorRow = np.fromfile(fileH,
                                    # dtype=get_type[elemSize],
                                    dtype = np.float32,
                                    # dtype = np.double,
                                    count = 8)
                row = np.concatenate((timestamp,errorRow))
                print(timestamp)
                print(ArUcoId)
                print(errorRow)
                # print(ArUcoTab)
                if ArUcoId in ArUcoTab.keys():
                    ArUcoTab[ArUcoId] = np.r_[ArUcoTab[ArUcoId], row.reshape((1,9))]
                else:
                    ArUcoTab[ArUcoId] = row.reshape((1,9))
        # return
        #  plot errors
        fig, ax = plt.subplots()
        fig.suptitle("Feature errors")
        for ki in ArUcoTab.keys():
            symbols = plot_time(ax, ArUcoTab[ki][:,0],ArUcoTab[ki][:,1:].T)

        # ylimits = [.0,10.1]
        # plt.ylim((ylimits[0],ylimits[1]))
        # labels = ["0","1","2","3","4","5"]
        # fig.legend(symbols,labels, loc=2)

        plt.tight_layout()
        plt.savefig(directory + str(i)+"/errores.pdf",bbox_inches='tight')
        #plt.show()
        plt.close()


        #
        # ####   Plot
        # #   Space errors
        # new_agents = []
        # end_position = np.zeros((n_agents,6))
        # for i in range(n_agents):
        #     cam = cm.camera()
        #     end_position[i,:] = agents[i].camera.p.copy()
        #     new_agents.append(ctr.agent(cam,pd[:,i],end_position[i,:],P))
        # if set_consensoRef:
        #     state_err +=  error_state(pd,new_agents, gdl = gdl,
        #                               name = directory+"/3D_error")
        # else:
        #     state_err +=  error_state_equal(new_agents, gdl = gdl,
        #                                     name = directory+"/3D_error")
        #
        # logText = 'Errores de consenso iniciales\n'
        # logText += str(err_array[:,:,0])
        # logText += '\n' +str(err_array[:,:,0].max())
        #
        # logText += '\n' +"State error = "+str(state_err)
        # if FOVflag:
        #     logText += '\n' +"WARNING : Cammera plane hit scene points: "+str( depthFlags)
        # if (pos_arr[:,2,:] < 0.).any():
        #     logText += '\n' +"WARNING : Possible plane colition to ground"
        # logText += '\n' +"-------------------END------------------"
        #
        # write2log(logText+'\n'+'\n')
        #
        # # Colors setup
        # #n_colors = max(n_agents,2*n_points)
        # #colors = randint(0,255,3*n_colors)/255.0
        # #colors = colors.reshape((n_colors,3))
        # npzfile = np.load("general.npz")
        # colors = npzfile["colors"]
        #
        # #   Save 3D plot data:
        # np.savez(directory + "/data3DPlot.npz",
        #         P = P, pos_arr=pos_arr, p0 = p0,
        #         pd = pd, end_position = end_position )
        #
        # if not enablePlot:
        #     return [ret_err, state_err, FOVflag]
        #
        # #   Camera positions in X,Y
        # mp.plot_position(pos_arr,
        #                 pd,
        #                 lfact = 1.1,
        #                 #colors = colors,
        #                 name = directory+"/Cameras_trayectories")
        #
        # #   3D plots
        # fig, ax = plt.subplots(ncols = 2,
        #                        frameon=False,
        #                        figsize=(8,6),
        #                         gridspec_kw={'width_ratios': [3,1]})
        # #fig = plt.figure(frameon=False, figsize=(5,3))
        # ax[0].axis('off')
        # ax[0] = fig.add_subplot(1, 2, 1, projection='3d')
        # name = directory+"/3Dplot"
        # #axplot(P[0,:], P[1,:], P[2,:], 'o')
        # ax[0].scatter(P[0,:], P[1,:], P[2,:], s = 20)
        # if gdl ==1:
        #     for i in range(n_agents):
        #         plot_3Dcam(ax[0], agents[i].camera,
        #                     pos_arr[i,:,:],
        #                     p0[:,i],
        #                     end_position[i,:],
        #                     pd[:,i],
        #                     color = colors[i],
        #                     i = i,
        #                     camera_scale    = 0.06)
        # elif gdl == 2:
        #     for i in range(n_agents):
        #         init = p0[:,i]
        #         end = end_position[i,:]
        #         ref = pd[:,i]
        #         init[3] = end[3] = ref[3] = pi
        #         init[4] = end[4] = ref[4] = 0
        #         plot_3Dcam(ax[0], agents[i].camera,
        #                     pos_arr[i,:,:],
        #                     init,
        #                     end,
        #                     ref,
        #                     color = colors[i],
        #                     i = i,
        #                     camera_scale    = 0.02)
        #
        # #   Plot cammera position at intermediate time
        # if midMarker:
        #     step_sel = int(pos_arr.shape[2]/2)
        #     ax[0].scatter(pos_arr[:,0,step_sel],
        #             pos_arr[:,1,step_sel],
        #             pos_arr[:,2,step_sel],
        #                 marker = '+',s = 200, color = 'black')
        #
        # lfact = 1.1
        # x_min = min(p0[0,:].min(),
        #             pd[0,:].min(),
        #             pos_arr[:,0,-1].min(),
        #             P[0,:].min())
        # x_max = max(p0[0,:].max(),
        #             pd[0,:].max(),
        #             pos_arr[:,0,-1].max(),
        #             P[0,:].max())
        # y_min = min(p0[1,:].min(),
        #             pd[1,:].min(),
        #             pos_arr[:,1,-1].min(),
        #             P[1,:].min())
        # y_max = max(p0[1,:].max(),
        #             pd[1,:].max(),
        #             pos_arr[:,1,-1].max(),
        #             P[1,:].max())
        # z_max = max(p0[2,:].max(),
        #             pd[2,:].max(),
        #             pos_arr[:,2,-1].max(),
        #             P[2,:].max())
        # z_min = min(p0[2,:].min(),
        #             pd[2,:].min(),
        #             pos_arr[:,2,-1].min(),
        #             P[2,:].min())
        #
        # width = x_max - x_min
        # height = y_max - y_min
        # depth = z_max - z_min
        # sqrfact = max(width,height,depth)
        #
        # x_min -= (sqrfact - width )/2
        # x_max += (sqrfact - width )/2
        # y_min -= (sqrfact - height )/2
        # y_max += (sqrfact - height )/2
        # z_min -= (sqrfact - depth )/2
        # z_max += (sqrfact - depth )/2
        # ax[0].set_xlim(x_min,x_max)
        # ax[0].set_ylim(y_min,y_max)
        # ax[0].set_zlim(z_min,z_max)
        #
        # #ax = fig.add_subplot(1, 2, 2)
        # symbols = []
        # labels = [str(i) for i in range(n_agents)]
        # for i in range(n_agents):
        #     symbols.append(mpatches.Patch(color=colors[i%colors.shape[0]]))
        # ax[1].legend(symbols,labels, loc=7)
        # ax[1].axis('off')
        #
        # #fig.legend( loc=1)
        # plt.savefig(name+'.pdf')#,bbox_inches='tight')
        # plt.close()
        #
        # #   Descriptores x agente
        # #       Predicted endpoints
        # pred = np.zeros((n_agents,2*n_points))
        # for i in range(n_agents):
        #     pred[i,:] = agents[i].s_ref.T.reshape(2*n_points) - desc_arr[i,:,0]
        # avrE = pred.mean(axis = 0)
        # for i in range(n_agents):
        #     pred[i,:] = desc_arr[i,:,0] + ( pred[i,:] - avrE)
        #
        # for i in range(n_agents):
        #     mp.plot_descriptors(desc_arr[i,:,:],
        #                         agents[i].camera.iMsize,
        #                         agents[i].s_ref,
        #                         #colors,
        #                         pred[i,:],
        #                         #enableLims = False,
        #                         name = directory+"/Image_Features_"+str(i),
        #                         label = "Image Features")
        #
        # #   Error de formación
        # mp.plot_time(t_array,
        #             serr_array[:,:],
        #             #colors,
        #             ylimits = [-0.1,1.1],
        #             ref = 0.1,
        #             refLab = "Threshold",
        #             name = directory+"/State_Error",
        #             label = "Formation Error",
        #             labels = ["translation","rotation"])
        #
        # #   Errores x agentes
        # for i in range(n_agents):
        #     mp.plot_time(t_array,
        #                 err_array[i,:,:],
        #                 #colors,
        #                 ylimits = [-1,1],
        #                 name = directory+"/Features_Error_"+str(i),
        #                 label = "Features Error")
        #
        # #   Errores x agentes
        # tmp = norm(err_array,axis = 1) / n_agents
        # mp.plot_time(t_array,
        #             tmp,
        #             #colors,
        #             ref = int_res,
        #             refLab = "Integral threshold",
        #             ylimits = [-1,1],
        #             name = directory+"/Norm_Feature_Error",
        #             label = "Features Error")
        #
        # #   Velocidaes x agente
        # for i in range(n_agents):
        #     mp.plot_time(t_array,
        #                 U_array[i,:,:],
        #                 #colors,
        #                 #ylimits = [-1,1],
        #                 ylimits = [-.1,.1],
        #                 name = directory+"/Velocidades_"+str(i),
        #                 label = "Velocities",
        #                 labels = ["X","Y","Z","Wx","Wy","Wz"])
        #
        # #   Posiciones x agente
        # for i in range(n_agents):
        #     mp.plot_time(t_array,
        #                 pos_arr[i,:3,:],
        #                 #colors,
        #                 name = directory+"/Traslaciones_"+str(i),
        #                 label = "Translations",
        #                 labels = ["X","Y","Z"])
        #     mp.plot_time(t_array,
        #                 pos_arr[i,3:,:],
        #                 #colors,
        #                 name = directory+"/Angulos_"+str(i),
        #                 label = "Angles",
        #                 labels = ["Roll","Pitch","yaw"],
        #                 module = [-pi,pi])
        #
        # #   Valores propios normal
        # for i in range(n_agents):
        #     mp.plot_time(t_array,
        #                 s_store[i,:,:],
        #                 #colors,
        #                 ylimits = [.0,10.1],
        #                 name = directory+"/ValoresPR_"+str(i),
        #                 label = "Non zero singular value magnitudes",
        #                 labels = ["0","1","2","3","4","5"])
        #
        # #   Valores propios inversa
        # for i in range(n_agents):
        #     mp.plot_time(t_array,
        #                 sinv_store[i,:,:],
        #                 #colors,
        #                 ylimits = [.0,10.1],
        #                 name = directory+"/ValoresP_"+str(i),
        #                 label = "Non zero singular value magnitudes",
        #                 labels = ["0","1","2","3","4","5"])
        #                 #limits = [[t_array[0],t_array[-1]],[0,20]])
        #
        #
        #
        # #   Proyección de error sobre vectores propios
        # #labels = [str(i) for i in range(2*n_points)]
        # labels =  ["0","1","2","3","4","5"]
        # for i in range(n_agents):
        #     mp.plot_time(t_array,
        #                 #svdProy[i,:,:],
        #                 svdProy[i,:6,:],
        #                 #colors,
        #                 name = directory+"/Proy_et_VH_"+str(i),
        #                 label = "Proy($e$) over $V^h$ ",
        #                 labels = labels)
        #                 #limits = [[t_array[0],t_array[-1]],[0,20]])

if __name__ ==  "__main__":

    main(sys.argv)
