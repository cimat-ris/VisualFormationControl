/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
March 1st, 2019
This code is used to declare some hand mande functions.
*/

/*********************************************************************************** c++ libraries */
#include <math.h>

/******************************************************************************** OpenCv libraries */
#include <opencv2/core.hpp>

using namespace cv;

Mat rotationZ(double yaw); //computes rotation in Z axis
Mat rotationY(double pitch); //computes rotation in Y axis
Mat rotationX(double roll); //computes rotation in X axis
Vec3d rotationMatrixToEulerAngles(Mat &R); //obtains euler angles from matrix
double distance(double *a, double *b, int n); //distance between two vectors
void normalize(double *a, int n); //normalizes a vector
