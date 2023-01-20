 
#include "vc_controller/vc_controller.h"

int chaumette( vcc::state & state,
               vcc::matching_result & Matching_result
              ){
    
    //  Normalización in situ
    vcc::camera_norm(state.params, Matching_result);
    
    //  Error
    cv::Mat err = Matching_result.p1-Matching_result.p2;
    
    //  estimación de Z
    int n = Matching_result.p2.rows;
    int type = Matching_result.p2.type();
    cv::Mat Z = cv::Mat::ones(n,1,type);
    Z = state.Z * Z;
    
    //  L = M_{s_i}
    cv::Mat L = vcc::interaction_Mat(Matching_result.p2,Z);
    if (L.empty())
        return -1;
    
    //  L = L^{-1}
    double det=0.0;
    L = vcc::Moore_Penrose_PInv(L,det);
    if (det < 1e-6)
        return -1;

    //  U = L^{-1} e 
    cv::Mat U = -1.0 *  L*err.reshape(1,L.cols); 
//     std::cout << U << std::endl << std::flush;
    
    
    /**********Updating velocities in the axis*/
    //velocities from homography decomposition
//     U = cv::Mat(U,CV_32F); // para generalizar el tipo de cariable
    state.Vx += (float) U.at<float>(1,0);
    state.Vy += (float) U.at<float>(0,0);
    state.Vz += (float) U.at<float>(2,0);
    state.Vroll += (float) U.at<float>(3,0);
    state.Vpitch += (float) U.at<float>(4,0);
    state.Vyaw += (float) U.at<float>(5,0);
    
    std::cout << state.Vx << ", " <<
    state.Vy << ", " <<
    state.Vz << ", " <<
    state.Vyaw << std::endl << std::flush;
    return 0;
}
