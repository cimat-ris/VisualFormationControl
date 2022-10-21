#include "vc_state/img_tools.h"

using namespace cv;

Mat Moore_Penrose_PInv(Mat L,double & det){
    
    Mat Lt = L.t();
    Mat Ls = Lt*L;
    det = determinant(Ls);
    if (det > 1e-6){
        return Ls.inv()*Lt;
    }
        
    return Lt;
}
