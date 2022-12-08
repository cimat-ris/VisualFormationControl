#include "Geometry.hpp"

/*
	function: rotationZ
	description: creates a matrix with rotation in Z axis
	params:
		yaw: angle in yaw
	returns: 
		Mat with the given yaw angle
*/
Mat rotationZ(double yaw){
	return (Mat_<double>(3, 3) <<                 
                  cos(yaw), -sin(yaw), 0,
                  sin(yaw),  cos(yaw), 0,
		  0,          0,           1);
}


/*
	function: rotationY
	description: creates a matrix with rotation in Y axis
	params:
		pitch: angle in pitch
	returns: 
		Mat with the given pitch angle
*/
Mat rotationY(double pitch){
	return (Mat_<double>(3, 3) <<
                  cos(pitch), 0, sin(pitch),
                  0, 1,          0,
                  -sin(pitch), 0,  cos(pitch));
}

/*
	function: rotationX
	description: creates a matrix with rotation in X axis
	params:
		roll: angle in roll
	returns;
		Mat with the given roll angle
*/
Mat rotationX(double roll){
	return (Mat_<double>(3, 3) <<
                  1,          0,           0,
                  0, cos(roll), -sin(roll),
                  0, sin(roll),  cos(roll));
}

/* 
	function: rotationMatrixToEulerAngles
	params: 
		R: rotation matrix
	returns: 
		vector containing the euler angles
	function taken from : https://www.learnopencv.com/rotation-matrix-to-euler-angles/
*/
Vec3d rotationMatrixToEulerAngles(Mat &R){      
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) ); 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular){
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }else{
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3d(x, y, z);        
}

/*
	function: distance
	description: computes the norm of the differences between two vectors
	params:	
		a: first vector
		b: second vector
		n: size of the vectors
	returns
		the norm of the difference
*/
double distance(double *a, double *b, int n){
	double d = 0;
	for(int i=0;i<n;i++)
		d+=(a[i]-b[i])*(a[i]-b[i]);
	
	return sqrt(d);
}

/*
	function: normalize
	description: normalizes a vector
	params:
		a: vector
		n: size of the vector
*/
void normalize(double *a, int n){
	double norm = 0;
	for(int i=0;i<n;i++)
		norm+=a[i]*a[i];

	norm = sqrt(norm);

	for(int i=0;i<n;i++)
		a[i]/=norm;
}
