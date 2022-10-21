#include "vc_state/img_tools.h"

using namespace cv;
 
Mat interaction_Mat(vc_homograpy_matching_result& result,
                double Z
                   ){
    
    int n = result.p2.rows;

    Mat L= Mat::zeros(n,12,CV_64F) ;
    
    //  Calculos
    //   -1/Z
    L.col(0) = -Mat::ones(n,1,CV_64F)/Z;
//     L.col(1) =
    //  p[0,:]/Z
    L.col(2) = result.p2.col(0)/Z;
    //  p[0,:]*p[1,:]
    L.col(3) = result.p2.col(0).mul(result.p2.col(1));
    //  -(1+p[0,:]**2)
    L.col(4) = -1.0*(1.0+result.p2.col(0).mul(result.p2.col(0)));
    //  p[1,:]
    result.p2.col(1).copyTo(L.col(5));
//     L.col(6) =
    //  -1/Z
    L.col(0).copyTo(L.col(7));
    //  p[1,:]/Z
    L.col(8) = result.p2.col(1)/Z;
    //  1+p[1,:]**2
    L.col(9) =  1.0+result.p2.col(1).mul(result.p2.col(1));
    //  -p[0,:]*p[1,:]
    L.col(10) = -1.0*result.p2.col(0).mul(result.p2.col(1));
    //  -p[0,:]
    L.col(11) = -1.0 * result.p2.col(0);

    return L.reshape(1,2*n);
}
