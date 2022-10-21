#include "vc_state/img_tools.h"

using namespace cv;

Mat Moore_Penrose_PInv(Mat L,double & det){
    
    std::cout << "DB2.2.1"<< std::endl;
//     std::cout << "L = " << L << std::endl;
    std::cout << "L.size = " << L.size() << std::endl;
    Mat Lt = L.t();
    Mat Ls = Lt*L;
    std::cout << "DB2.2.2"<< std::endl;
    det = determinant(Ls);
    std::cout << "DB2.2.3"<< std::endl;
    if (det > 1e-6){
        return Ls.inv()*Lt;
    }
    std::cout << "DB2.2.4"<< std::endl;
        
    return Lt;
}
