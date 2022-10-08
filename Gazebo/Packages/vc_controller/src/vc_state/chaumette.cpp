 
#include "vc_state/vc_state.h"

using namespace cv;
using namespace std;

int chaumette(Mat img,
               vc_state & state,
               vc_homograpy_matching_result & matching_result
//                sensor_msgs::ImagePtr image_msg
              ){
    cout << "DB1" << endl;
    
    if (compute_descriptors(img,state.params,state.desired_configuration,matching_result)<0)
      return -1;
    
    
//         
        cout << "DB2" << endl;
		// Descriptor control
        double lambda = 1.0;
        cout << "Before norm P1 " << matching_result.p1 << endl;
        cout << "Before norm P2 " << matching_result.p2 << endl;
        Mat err = matching_result.p1-matching_result.p2;
                cout << "err = " << err << endl;
       camera_norm(state.params, matching_result);
        cout << "After norm P1 " << matching_result.p1 << endl; // << matching_result.p2 << endl;
        cout << "After norm P2 " << matching_result.p2 << endl;
        err = matching_result.p1-matching_result.p2;
//         cout << "err = " << err << endl;
        cout << "DB2.1" << endl;
//         camera_norm(state.params, matching_result);
        Mat L = interaction_Mat(matching_result,1.0);
//         cout << "L  = \n" << L <<  endl;
        cout << "DB2.2" << endl;
        double det=0.0;
        L = Moore_Penrose_PInv(L,det);
        if (det < 1e-6)
            return -1;
        cout << "DB2.3" << endl;
        cout << "check before: " << err.at<double>(0,0) << " " << err.at<double>(0,1)
                    << " after: " << err.reshape(1,L.cols).at<double>(0,0) << " " << err.reshape(1,L.cols).at<double>(1,0) << endl;
//         cout << "err = " << err << endl;
//         cout << "After p " << matching_result.p1 << endl << matching_result.p2 << endl;
        Mat U = -1.0 * lambda * L*err.reshape(1,L.cols); 
        
        cout << "DB3" << endl;
        
        
		/**********Updating velocities in the axis*/
        //velocities from homography decomposition
//         cout << "L  = \n" << L <<  endl;
		state.Vx = (float) U.at<double>(0,0);
		state.Vy = (float) U.at<double>(1,0);
		state.Vz = (float) U.at<double>(2,0);
//         state.Vroll = (float) U.at<double>(3,0);
//         state.Vpitch = (float) U.at<double>(4,0);
// 		state.Vyaw = (float) U.at<double>(5,0);
		cout << "DB4" << endl;
		return 0;
}
