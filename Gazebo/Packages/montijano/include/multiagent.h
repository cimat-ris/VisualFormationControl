#ifndef MULTIAGENT
#define MULTIAGENT

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <montijano/geometric_constraint.h>

#include "utils_vc.h"
#include "utils_geom.h"
#include "utils_params.h"



class multiagent_state{
    
public:
    //  Variables
    //  TODO, accept n form global 
    int n = 3;
    double x_aster[3][3], y_aster[3][3],z_aster[3][3],yaw_aster[3][3]; //desired poses
    int d[3][3] = {0};
    double xc[3][3],yc[3][3],z[3][3],yaw[3][3];
    double  L[3][3] ;
    int n_neigh = 0;
    int neighbors[3]={0};
    int actual ;
    int info[3],rec[3]; //neighbors comunicating the image description and receiving homgraphy
    int n_info;
    
    // METHODS
    void load(const ros::NodeHandle &nh, std::string sel);
    void update(double rollj, double pitchj, const montijano_parameters&params, montijano_state &state ,  montijano_control &control, cv::Mat H, int ii, int jj);
};






#endif // MULTIAGENT
