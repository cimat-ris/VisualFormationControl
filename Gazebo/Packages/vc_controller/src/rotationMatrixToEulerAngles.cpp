#include "vc_controller/img_tools.h"

using namespace vcc;

/*
	function: rotationMatrixToEulerAngles
	params:
		R: rotation matrix
	result:
		vector containing the euler angles
	function taken from : https://www.learnopencv.com/rotation-matrix-to-euler-angles/
*/
cv::Vec3f vcc::rotationMatrixToEulerAngles(const cv::Mat & R){
    
    //  sy = sqrt(R[0,0]**2 +R[1,0]**2)
    float sy = R.at<double>(0,0);
    sy *= R.at<double>(0,0);
    sy +=  R.at<double>(1,0) * R.at<double>(1,0) ;
    sy = sqrt(sy);
    
    float x, y, z;
    //  If not singular
    if (sy >= 1e-6){
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }else{
        //  IF Singular
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
}
