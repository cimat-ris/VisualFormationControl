#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <ros/ros.h>


class vc_parameters {
	public:
		// Control parameters
		float Kv;
		float Kw;
		float dt;

		// Image proessing parameters
		float feature_threshold;
		int nfeatures=250;
		float scaleFactor=1.2f;
		int nlevels=8;
		int edgeThreshold=15; // Changed default (31);
		int firstLevel=0;
		int WTA_K=2;
		cv::ORB::ScoreType scoreType=cv::ORB::HARRIS_SCORE;
		int patchSize=31;
		int fastThreshold=20;

		// Camera parameters
		cv::Mat K;

		// Methods
		vc_parameters();
		void load(const ros::NodeHandle &nh);
};
