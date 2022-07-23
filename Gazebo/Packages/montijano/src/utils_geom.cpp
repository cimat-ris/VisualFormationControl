#include "utils_geom.h"

using namespace cv;

/*
	function: rotationMatrixToEulerAngles
	params:
		R: rotation matrix
	result:
		vector containing the euler angles
	function taken from : https://www.learnopencv.com/rotation-matrix-to-euler-angles/
*/
Vec3f rotationMatrixToEulerAngles(const Mat &R){
		float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
		bool singular = sy < 1e-6; // If
		float x, y, z;
		if (!singular){
				x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
				y = atan2(-R.at<double>(2,0), sy);
				z = atan2(R.at<double>(1,0), R.at<double>(0,0));
		}else{
				x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
				y = atan2(-R.at<double>(2,0), sy);
				z = 0;
		}
		return Vec3f(x, y, z);
}

Mat rotationX(double roll){
	return (Mat_<double>(3, 3) <<
                  1,          0,           0,
                  0, cos(roll), -sin(roll),
                  0, sin(roll),  cos(roll));
}

/*
	function: rotationY
	description: creates a matrix with rotation in Y axis
	params:
		pitch: angle in pitch
*/
Mat rotationY(double pitch){
	return (Mat_<double>(3, 3) <<
                  cos(pitch), 0, sin(pitch),
                  0, 1,          0,
                  -sin(pitch), 0,  cos(pitch));
}

/*
	function: rotationZ
	description: creates a matrix with rotation in Z axis
	params:
		yaw: angle in yaw
*/
Mat rotationZ(double yaw){
	return (Mat_<double>(3, 3) <<                 
                  cos(yaw), -sin(yaw), 0,
                  sin(yaw),  cos(yaw), 0,
		  0,          0,           1);
}
