#include <opencv2/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/calib3d.hpp>

cv::Vec3f rotationMatrixToEulerAngles(const cv::Mat &R);
