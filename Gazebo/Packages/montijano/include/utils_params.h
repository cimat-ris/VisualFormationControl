#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <ros/ros.h>

#ifndef UTILS_PARAMS
#define UTILS_PARAMS

class vc_parameters {
	public:
		// Control parameters
		float Kv;
		float Kw;
		float dt;

		// Image proessing parameters
		float feature_threshold;
		int nfeatures;
		float scaleFactor;
		int nlevels;
		int edgeThreshold; // Changed default (31);
		int firstLevel;
		int WTA_K;
		cv::ORB::ScoreType scoreType;
		int patchSize;
		int fastThreshold;
    float flann_ratio;
		// Camera parameters
		cv::Mat K;

		// Methods
		vc_parameters();
		void load(const ros::NodeHandle &nh);
};

#endif
