 
#include "vc_state/vc_state.h"

using namespace cv;
using namespace std;

int chaumette(Mat img,
               vc_state & state,
               vc_homograpy_matching_result & matching_result
              ){
    cout << "DB1" << endl;
    
    if (compute_descriptors(img,state.params,state.desired_configuration,matching_result)<0)
      return -1;
    
    
//         
		// Descriptor control
        double lambda = 1.0;
        Mat err = matching_result.p1-matching_result.p2;
       camera_norm(state.params, matching_result);
        err = matching_result.p1-matching_result.p2;
        Mat L = interaction_Mat(matching_result,1.0);
        double det=0.0;
        L = Moore_Penrose_PInv(L,det);
        if (det < 1e-6)
            return -1;

        Mat U = -1.0 * lambda * L*err.reshape(1,L.cols); 
        
        
		/**********Updating velocities in the axis*/
        //velocities from homography decomposition
		state.Vx = (float) U.at<double>(1,0);
		state.Vy = (float) U.at<double>(0,0);
		state.Vz = (float) U.at<double>(2,0);
//         state.Vroll = (float) U.at<double>(3,0);
//         state.Vpitch = (float) U.at<double>(4,0);
		state.Vyaw = (float) U.at<double>(5,0);
		return 0;
}
