#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "utils_params.h"
#include "utils_vc.h"

class vc_homograpy_matching_result {
	public:
		cv::Mat H;
		cv::Mat img_matches;
		std::vector<cv::Point2f> p1;
		std::vector<cv::Point2f> p2;
		double mean_feature_error;
		vc_homograpy_matching_result();
};

int compute_homography(const cv::Mat&, const vc_parameters&, const vc_desired_configuration&,vc_homograpy_matching_result &);
int select_decomposition(const std::vector<cv::Mat> &Rs,
												const std::vector<cv::Mat> &Ts,
												const std::vector<cv::Mat> &Ns,
												const vc_homograpy_matching_result& matching_result,
												int &selected,cv::Mat&Rbest,cv::Mat &tbest);
