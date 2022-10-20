#include "vc_state/img_tools.h"

using namespace cv;
 
Mat interaction_Mat(vc_homograpy_matching_result& result,
                double Z
                   ){
    
    Mat p1 = Mat(result.p1); Mat p2 = Mat(result.p2);
    
    int n = p2.rows;
//     std::cout << "interaction_Mat: p2.size  = " << p2.size << std::endl;
//     std::cout << "p2 = " << p2 << std::endl;
//     std::cout << "DB2.1.a" << std::endl;
//     camera_norm(params, result);
//     std::cout << "DB2.1.b" << std::endl;
    Mat L= Mat::zeros(n,12,CV_64F) ;
    
    //  Calculos
//     std::cout << "DB2.1.1" << std::endl;
    //   -1/Z
    L.col(0) = -Mat::ones(n,1,CV_64F)/Z;
//     L.col(1) =
    //  p[0,:]/Z
//     std::cout << "DB2.1.2" << std::endl;
    L.col(2) = p2.col(0)/Z;
    //  p[0,:]*p[1,:]
//     std::cout << "DB2.1.3" << std::endl;
    L.col(3) = p2.col(0).mul(p2.col(1));
    //  -(1+p[0,:]**2)
    L.col(4) = -1.0*(1.0+p2.col(0).mul(p2.col(0)));
    //  p[1,:]
//     std::cout << "DB2.1.4" << std::endl;
    p2.col(1).copyTo(L.col(5));
//     L.col(6) =
    //  -1/Z
    L.col(0).copyTo(L.col(7));
    //  p[1,:]/Z
//     std::cout << "DB2.1.5" << std::endl;
    L.col(8) = p2.col(1)/Z;
    //  1+p[1,:]**2
    L.col(9) =  1.0+p2.col(1).mul(p2.col(1));
    //  -p[0,:]*p[1,:]
//     std::cout << "DB2.1.6" << std::endl;
    L.col(10) = -1.0*p2.col(0).mul(p2.col(1));
    //  -p[0,:]
//     p2.col(0).copyTo(L.col(11));
    L.col(11) = -1.0 * p2.col(0);
//     std::cout << "L = " << L << std::endl;
    return L.reshape(1,2*n);
}
