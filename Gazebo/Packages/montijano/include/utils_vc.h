#include <iostream>
#include <string>
#include <vector>
#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "utils_params.h"
#include <Eigen/Dense>

#ifndef UTILS_VC
#define UTILS_VC

class vc_desired_configuration {
  public:
    cv::Mat descriptors;
    std::vector<cv::KeyPoint> kp;
    cv::Mat img;
};

class vc_control {
	public:
		float Vx,Vy,Vz,Vyaw;
		vc_control() : Vx(0),Vy(0),Vz(0),Vyaw(0) {}
};

class vc_state {
	public:
		/* defining where the drone will move and integrating system*/
		float X,Y,Z,Yaw,Pitch,Roll;
		bool initialized;
		float t,dt;
		float Kv,Kw;
		ros::Publisher ros_pub;
		// Methods
		vc_state();
		inline void set_gains(const vc_parameters&params) {
			this->Kv = params.Kv;
			this->Kw = params.Kw;
			this->dt = params.dt;
		};
		std::pair<Eigen::VectorXd,float> update(const vc_control &command);
		void initialize(const float &x,const float &y,const float &z,const float &yaw);
};

#endif
