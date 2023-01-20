#include "vc_controller/img_tools.h"

using namespace vcc;
 
void vcc::camera_norm(const parameters & params, 
                       matching_result& result){
        //  Normalización in situ
    
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
