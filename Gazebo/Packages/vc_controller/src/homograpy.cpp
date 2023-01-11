 
#include "vc_controller/vc_controller.h"


// int homography(cv::Mat img,
int homography(
               vcc::state & State,
               vcc::homograpy_matching_result & matching_result
              ){
    
//     int test = vcc::compute_homography(
//         img,State.params,
//         State.Desired_Configuration,
//         matching_result);
//     if (test<0)
//         return -1;

    // Decompose homography*/
    std::vector<cv::Mat> Rs;
    std::vector<cv::Mat> Ts;
    std::vector<cv::Mat> Ns;
    cv::decomposeHomographyMat(matching_result.H,State.params.K,Rs,Ts,Ns);

    // Select decomposition
    vcc::select_decomposition(
        Rs,Ts,Ns,
        matching_result,
        State.selected,
        State.R_best,
        State.t_best);
    
    /**********Computing velocities in the axis*/
    //velocities from homography decomposition
    State.Vx += (float) State.t_best.at<double>(0,1);
    State.Vy += (float) State.t_best.at<double>(0,0);
    State.Vz += (float) State.t_best.at<double>(0,2); //due to camera framework
    //velocities from homography decomposition and euler angles.
    cv::Vec3f angles = vcc::rotationMatrixToEulerAngles(State.R_best);
    State.Vyaw += (float) -angles[2];//due to camera framework
    return 0;
}
