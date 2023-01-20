#include "vc_controller/img_tools.h"

using namespace vcc;
 
cv::Mat vcc::interaction_Mat(cv::Mat & p, cv::Mat & Z ){
    
    int n = p.rows;
    int type = p.type();
    
    if (n != Z.rows)
    {
        std::cout << "Z vector mismatches dimetions \n" <<  std::flush;
        return cv::Mat();
    }
//     cv::Mat L= cv::Mat::zeros(n,12,CV_64F) ;
    cv::Mat L= cv::Mat::zeros(n,12,type) ;
    
    //  Calculos
    //   -1/Z
//     L.col(0) = - cv::Mat::ones(n,1,CV_64F)/Z;
    L.col(0) = - cv::Mat::ones(n,1,type)/Z;
//     L.col(1) =
    //  p[0,:]/Z
    L.col(2) = p.col(0)/Z;
    //  p[0,:]*p[1,:]
    L.col(3) = p.col(0).mul(p.col(1));
    //  -(1+p[0,:]**2)
    L.col(4) = -1.0*(1.0+p.col(0).mul(p.col(0)));
    //  p[1,:]
    p.col(1).copyTo(L.col(5));
//     L.col(6) =
    //  -1/Z
    L.col(0).copyTo(L.col(7));
    //  p[1,:]/Z
    L.col(8) = p.col(1)/Z;
    //  1+p[1,:]**2
    L.col(9) =  1.0+p.col(1).mul(p.col(1));
    //  -p[0,:]*p[1,:]
    L.col(10) = -1.0*p.col(0).mul(p.col(1));
    //  -p[0,:]
    L.col(11) = -1.0 * p.col(0);

    return L.reshape(1,2*n);
}
