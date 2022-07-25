 
#include "multiagent.h"

using namespace cv;
using namespace std;

 void multiagent_state::load(const ros::NodeHandle &nh, string sel){
     
     // Load desired poses
	XmlRpc::XmlRpcValue kConfig;
	string param_name = "desired_poses_"+sel;
    if (nh.hasParam(param_name)) {
			nh.getParam(param_name, kConfig);
			if (kConfig.getType() == XmlRpc::XmlRpcValue::TypeArray)
			for (int i=0;i<9;i++) {
						 std::ostringstream ostr;
						 ostr << kConfig[i];
						 std::istringstream istr(ostr.str());
						 istr >> this->x_aster[i/3][i%3];
			}
			for (int i=0;i<9;i++) {
						 std::ostringstream ostr;
						 ostr << kConfig[i];
						 std::istringstream istr(ostr.str());
						 istr >> this->y_aster[i/3][i%3];
			}
			for (int i=0;i<9;i++) {
						 std::ostringstream ostr;
						 ostr << kConfig[i];
						 std::istringstream istr(ostr.str());
						 istr >> this->z_aster[i/3][i%3];
			}
	}
	cout << "[INF] Calibration Matrix " << endl << this->x_aster << endl;
     
}
 
 void multiagent_state::update(double rollj, double pitchj, const montijano_parameters&params, montijano_state &state ,  montijano_control &control, cv::Mat H, int ii, int jj){

     
    Mat RXj = rotationX(rollj);
    Mat RXi = rotationX(state.Roll);
    Mat RYj = rotationY(pitchj);
    Mat RYi = rotationY(state.Pitch);
    //rectify homography
    Mat Hir = RXi.inv() * RYi.inv()*params.K.inv();
    Mat Hjr = RXj.inv() * RYj.inv()*params.K.inv();
    Mat Hr = Hir*H*Hjr.inv();
     
    if(this->d[jj][ii] == 0){
        this->d[jj][ii]  =1;
        this->xc[jj][ii] = Hr.at<double>(1,2);
        this->yc[jj][ii] = Hr.at<double>(0,2);
        //velocities from homography 		
        control.Vx += params.Kv*(state.Z*Hr.at<double>(1,2)-this->x_aster[jj][ii]);
        control.Vy +=  params.Kv*(state.Z*Hr.at<double>(0,2)-this->y_aster[jj][ii]);		
        control.Vyaw += params.Kw*(atan2(Hr.at<double>(1,0),Hr.at<double>(0,0))-this->yaw_aster[jj][ii]);
    }else{

        vector<Mat> rotations;
        vector<Mat> translations;
        vector<Mat> normals;
        decomposeHomographyMat(H, params.K, rotations,translations, normals);

        double gg2[2],min=1e10;
        for(int i =0;i<4;i++){
            double c[2],c2[2];
            for(int j=0;j<2;j++){
                c[j] = translations[i].at<double>(j,0);			
            }c2[0] = this->xc[jj][ii]; c2[1] =this->yc[jj][ii]; 
            
            double aux = c[1];
            c[1] = c[0];
            c[0] = aux;
            double k = sqrt((c[0]-c2[0])*(c[0]-c2[0])+(c[1]-c2[1])*(c[1]-c2[1]));	
            if(k < min){
                min = k;
                gg2[0] = c[0]; gg2[1]=c[1];
            }
        }
        this->xc[jj][ii] = gg2[0];
        this->yc[jj][ii] = gg2[1];
        control.Vx += params.Kv*(state.Z*xc[jj][ii]-this->x_aster[jj][ii]);
        control.Vy +=  params.Kv*(state.Z*yc[jj][ii]-this->y_aster[jj][ii]);	
        control.Vyaw += params.Kw*(atan2(Hr.at<double>(1,0),Hr.at<double>(0,0))-this->yaw_aster[jj][ii]);
    }
}
