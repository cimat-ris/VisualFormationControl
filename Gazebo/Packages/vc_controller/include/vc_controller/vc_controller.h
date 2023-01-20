#ifndef VC_CONTROLLER_H
#define VC_CONTROLLER_H
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <Eigen/Dense>
#include "vc_controller/img_tools.h"
// #include "vc_state/math_custom.h"

namespace vcc
{
class state {
	public:
		/* defining where the drone will move and integrating system*/
		float X= 0.0,Y= 0.0,Z= 0.0,Yaw= 0.0,Pitch= 0.0,Roll= 0.0;
		bool initialized=false;
        
        /* Control parameters  */
        float Vx=0.0,Vy=0.0,Vz=0.0;
        float Vyaw=0.0, Vroll = 0.0, Vpitch = 0.0;
        float Kv=1.0;
        float Kw=1.0;
        float dt=0.025;
        
        // Image proessing parameters
        parameters params;
        //  Desired configuration
        desired_configuration Desired_Configuration;
        
        //  Best approximations
        bool selected = false;
        cv::Mat t_best;
        cv::Mat R_best; //to save the rot and trans
        
        // Methods
		state();

        std::pair<Eigen::VectorXd,float> update();
        void load(const ros::NodeHandle &nh);
		void initialize(const float &x,
                        const float &y,
                        const float &z,
                        const float &yaw);
        void save_data(double time, std::string directory);

};


}
//  TODO: propuesta: pasar los controladores dentro del namespace
//          que se defina el proceso de selección de control en el nodo
//  Controllers

// int homography(cv::Mat img,
//                vcc::state & state,
//                vcc::matching_result & matching_result
//               );
// 
// int chaumette(cv::Mat img,
//                vcc::state & state,
//                vcc::matching_result & matching_result
//               );
int homography(vcc::state & state,
               vcc::matching_result & matching_result
              );

int chaumette(vcc::state & state,
               vcc::matching_result & matching_result
              );

// Controller selection array only for vc_controller.h
#ifdef VC_CONTROLLER_NODE_H
// typedef int (*funlist) (cv::Mat img,
//                         vcc::state & state,
//                         vcc::matching_result & matching_result
//                        );
// funlist controllers[] = {&homography,&chaumette};

typedef int (*contList) (vcc::state & state,
                        vcc::matching_result & matching_result
                       );
const contList controllers[] = {&homography,&chaumette};

typedef int (*prepList) (const cv::Mat&img,
        const vcc::parameters & params, 
        const vcc::desired_configuration & desired_configuration,
        vcc::matching_result& result);
const prepList preprocessors[] = {&vcc::compute_homography,
                            &vcc::compute_descriptors};
#endif

#endif
