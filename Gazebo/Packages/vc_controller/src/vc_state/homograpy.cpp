 
#include "vc_state/vc_state.h"

using namespace cv;
using namespace std;

int homography(Mat img,
               vc_state & state,
               vc_homograpy_matching_result & matching_result,
               sensor_msgs::ImagePtr image_msg
              ){
    
    if (compute_homography(img,state.params,state.desired_configuration,matching_result)<0)
      return -1;
		/************************************************************* Prepare message */
		image_msg = cv_bridge::CvImage(std_msgs::Header(),sensor_msgs::image_encodings::BGR8,matching_result.img_matches).toImageMsg();
		image_msg->header.frame_id = "matching_image";
		 image_msg->width = matching_result.img_matches.cols;
		image_msg->height = matching_result.img_matches.rows;
		image_msg->is_bigendian = false;
		image_msg->step = sizeof(unsigned char) * matching_result.img_matches.cols*3;
		image_msg->header.stamp = ros::Time::now();

		// Decompose homography*/
		vector<Mat> Rs;
		vector<Mat> Ts;
		vector<Mat> Ns;
		decomposeHomographyMat(matching_result.H,state.params.K,Rs,Ts,Ns);

        // Select decomposition
        
        select_decomposition(Rs,Ts,Ns,matching_result,state.selected,state.R_best,state.t_best);

		/**********Computing velocities in the axis*/
		//velocities from homography decomposition
		state.Vx = (float) state.t_best.at<double>(0,1);
		state.Vy = (float) state.t_best.at<double>(0,0);
		state.Vz = (float) state.t_best.at<double>(0,2); //due to camera framework
		//velocities from homography decomposition and euler angles.
		Vec3f angles = rotationMatrixToEulerAngles(state.R_best);
		state.Vyaw = (float) -angles[2];//due to camera framework
		
		return 0;
}
