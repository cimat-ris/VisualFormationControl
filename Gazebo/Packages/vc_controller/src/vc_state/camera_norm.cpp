#include "vc_state/img_tools.h"

using namespace cv;
 
void camera_norm(const vc_parameters & params, 
                       vc_homograpy_matching_result& result){
        //  Normalización in situ
    int n = result.p1.rows;
    //  p1
    result.p1.col(0) = result.p1.col(0)-params.K.at<double>(0,2);
    result.p1.col(1) = result.p1.col(1)-params.K.at<double>(1,2);
    result.p1.col(0) = result.p1.col(0).mul(1.0/params.K.at<double>(0,0));
    result.p1.col(1) = result.p1.col(1).mul(1.0/params.K.at<double>(1,1));
    
    //  p2
    result.p2.col(0) = result.p2.col(0)-params.K.at<double>(0,2);
    result.p2.col(1) = result.p2.col(1)-params.K.at<double>(1,2);
    result.p2.col(0) = result.p2.col(0).mul(1.0/params.K.at<double>(0,0));
    result.p2.col(1) = result.p2.col(1).mul(1.0/params.K.at<double>(1,1));
   

    return;
}
