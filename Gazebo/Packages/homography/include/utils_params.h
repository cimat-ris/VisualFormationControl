#include <opencv2/core.hpp>
#include <ros/ros.h>


class vc_parameters {
	public:
		// Control parameters
		float Kv;
		float Kw;
		float dt;

		// Image proessing parameters
		float feature_threshold;

		// Camera parameters
		cv::Mat K;

		void load(const ros::NodeHandle &nh);

};
