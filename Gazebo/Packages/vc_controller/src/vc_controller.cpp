#include "vc_controller/vc_controller.h" 
using namespace vcc;

vcc::state::state() {
    X=0;
    Y=0;
    Z=0;
    
    Yaw=0;
    Pitch=0;
    Roll=0;
    initialized=false;
    dt=0.025;
    Kv=1.;
    Kw=1.;
}
                    
void vcc::state::load(const ros::NodeHandle &nh) {
    // Load intrinsic parameters
    XmlRpc::XmlRpcValue kConfig;
    this->params.K = cv::Mat(3,3, CV_64F, double(0));
    if (nh.hasParam("camera_intrinsic_parameters")) {
        nh.getParam("camera_intrinsic_parameters", kConfig);
        if (kConfig.getType() == XmlRpc::XmlRpcValue::TypeArray)
        for (int i=0;i<9;i++) {
            std::ostringstream ostr;
            ostr << kConfig[i];
            std::istringstream istr(ostr.str());
            istr >> this->params.K.at<double>(i/3,i%3);
        }
    }
// 	cout << "[INF] Calibration Matrix " << endl << this->params.K << endl;
    // Load error threshold parameter
    this->params.feature_threshold=nh.param(std::string("feature_error_threshold"),std::numeric_limits<double>::max());
    // Load feature detector parameters
    this->params.nfeatures=nh.param(std::string("nfeatures"),100);
    this->params.scaleFactor=nh.param(std::string("scaleFactor"),1.0);
    this->params.nlevels=nh.param(std::string("nlevels"),5);
    this->params.edgeThreshold=nh.param(std::string("edgeThreshold"),15);
    this->params.patchSize=nh.param(std::string("patchSize"),30);
    this->params.fastThreshold=nh.param(std::string("fastThreshold"),20);
    this->params.flann_ratio=nh.param(std::string("flann_ratio"),0.7);

    // Load gain parameters
    this->Kv=nh.param(std::string("gain_v"),0.0);
    this->Kw=nh.param(std::string("gain_w"),0.0);

    // Load sampling time parameter
    this->dt=nh.param(std::string("dt"),0.01);
}

std::pair<Eigen::VectorXd,float> vcc::state::update() {
    // Integrating
    this->X = this->X + this->Kv*this->Vx*this->dt;
    this->Y = this->Y + this->Kv*this->Vy*this->dt;
    this->Z = this->Z + this->Kv*this->Vz*this->dt;
    this->Yaw = this->Yaw + this->Kw*this->Vyaw*this->dt;

    Eigen::VectorXd position; 
    position.resize(3);
    position(0) = this->X;
    position(1) = this->Y; 
    position(2) = this->Z;

    return std::make_pair(position,this->Yaw);
}

void vcc::state::initialize(
    const float &x,const float &y,const float &z,
    const float &yaw) {
    this->X = x;
    this->Y = y;
    this->Z = z;
    this->Yaw = yaw;
    this->initialized = true;
  
}

//  Appends state data to spacified directory with the given timestamp
void vcc::state::save_data(double time, std::string directory)
{
    std::ofstream outfile(directory, std::ios::app | std::ios::binary);
    
    outfile.write((char *) & time,sizeof(double));
    outfile.write((char *) & X,sizeof(float));
    outfile.write((char *) & Y,sizeof(float));
    outfile.write((char *) & Z,sizeof(float));
    outfile.write((char *) & Yaw,sizeof(float));
    outfile.write((char *) & Pitch,sizeof(float));
    outfile.write((char *) & Roll,sizeof(float));
    outfile.write((char *) & Vx,sizeof(float));
    outfile.write((char *) & Vy,sizeof(float));
    outfile.write((char *) & Vz,sizeof(float));
    outfile.write((char *) & Vyaw,sizeof(float));
    outfile.write((char *) & Vpitch,sizeof(float));
    outfile.write((char *) & Vroll,sizeof(float));

    
    outfile.close();
}
