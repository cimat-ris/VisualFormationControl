 
#include "vc_controller/vc_controller.h"

// int chaumette(cv::Mat img,
int chaumette(
               vcc::state & state,
               vcc::homograpy_matching_result & matching_result
              ){
    
    
    //  Descriptor calculation, Halt if none are found
//     int test = vcc::compute_descriptors(
//         img,
//         state.params,
//         state.Desired_Configuration,
//         matching_result);
//     if (test<0)
//         return -1;
    
    // Descriptor control
    double lambda = 1.0;
    cv::Mat err = matching_result.p1-matching_result.p2;
    vcc::camera_norm(state.params, matching_result);
    err = matching_result.p1-matching_result.p2;
    cv::Mat L = vcc::interaction_Mat(matching_result,1.0);
    double det=0.0;
    L = vcc::Moore_Penrose_PInv(L,det);
    if (det < 1e-6)
        return -1;

    cv::Mat U = -1.0 * lambda * L*err.reshape(1,L.cols); 
    std::cout << U << std::endl << std::flush;
    
    
    /**********Updating velocities in the axis*/
    //velocities from homography decomposition
    //  TODO: U = cv::Mat(U,CV_32F); // para generalizar el tipo de cariable
    state.Vx += (float) U.at<float>(1,0);
    state.Vy += (float) U.at<float>(0,0);
    state.Vz += (float) U.at<float>(2,0);
//         state.Vroll = (float) U.at<float>(3,0);
//         state.Vpitch = (float) U.at<float>(4,0);
    state.Vyaw += (float) U.at<float>(5,0);
    
    std::cout << state.Vx << ", " <<
    state.Vy << ", " <<
    state.Vz << ", " <<
    state.Vyaw << std::endl << std::flush;
    return 0;
}
