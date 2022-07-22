#include <iostream>
#include <string>
#include <vector>
#include <ros/ros.h>

#include "utils_params.h"
#include <Eigen/Dense>

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
