#include "utils_params.h"
#include <iostream>

using namespace cv;
using namespace std;

void vc_parameters::load(const ros::NodeHandle &nh) {
	// Load intrinsic parameters
	XmlRpc::XmlRpcValue kConfig;
	this->K = Mat(3,3, CV_64F, double(0));
	if (nh.hasParam("camera_intrinsic_parameters")) {
			nh.getParam("camera_intrinsic_parameters", kConfig);
			if (kConfig.getType() == XmlRpc::XmlRpcValue::TypeArray)
			for (int i=0;i<9;i++) {
						 std::ostringstream ostr;
						 ostr << kConfig[i];
						 std::istringstream istr(ostr.str());
						 istr >> this->K.at<double>(i/3,i%3);
			}
	}
	cout << "[INF] Calibration Matrix " << endl << this->K << endl;
	// Load error threshold parameter
	this->feature_threshold=nh.param(std::string("feature_error_threshold"),std::numeric_limits<double>::max());
	// Load gain parameters
	this->Kv=nh.param(std::string("gain_v"),0.0);
	this->Kw=nh.param(std::string("gain_w"),0.0);
	// Load sampling time parameter
	this->dt=nh.param(std::string("dt"),0.01);
}
