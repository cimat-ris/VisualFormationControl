#include "vc_controller/img_tools.h"

using namespace vcc;

cv::Mat vcc::Moore_Penrose_PInv(cv::Mat L,double & det){
    
    cv::Mat Lt = L.t();
    cv::Mat Ls = Lt*L;
    det = cv::determinant(Ls);
    if (det > 1e-6){
        return Ls.inv()*Lt;
    }
        
    return Lt;
}
