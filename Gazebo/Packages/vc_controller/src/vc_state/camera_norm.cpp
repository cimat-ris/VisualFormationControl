#include "vc_state/img_tools.h"

using namespace cv;
 
void camera_norm(const vc_parameters & params, 
                       vc_homograpy_matching_result& result){
    
        //  Normalización in situ
    int n = result.p1.size();
    //  p1
    for(int i = 0; i < n ; i++){
        
    //     Mat tmp =  p1[i][0]-params.K.at<double>(0,2);
        Point2f tmp(params.K.at<double>(0,2),params.K.at<double>(1,2));
        result.p1[i] = result.p1[i]-tmp;
        result.p1[i] = result.p1[i]/params.K.at<double>(0,0);
        
        result.p2[i] = result.p2[i]-tmp;
        result.p2[i] = result.p2[i]/params.K.at<double>(0,0);
//         result.p1[i][1] = result.p1[i][1]-params.K.at<double>(1,2);
//         result.p1[i][0] = result.p1[i][0]/params.K.at<double>(0,0);
//         result.p1[i][1] = result.p1[i][1]/params.K.at<double>(1,1);
//         
//         //  result.p2
//         result.p2[i][0] = result.p2[i][0]-params.K.at<double>(0,2);
//         result.p2[i][1] = result.p2[i][1]-params.K.at<double>(1,2);
//         result.p2[i][0] = result.p2[i][0]/params.K.at<double>(0,0);
//         result.p2[i][1] = result.p2[i][1]/params.K.at<double>(1,1);
    }
//     Mat p1 = Mat(result.p1); Mat p2 = Mat(result.p2);
//     
//         //  Normalización in situ
//     int n = p1.rows;
//     //  p1
//     std::cout << "Camera norm: p1.size  = " << p1.size << std::endl;
//     std::cout << "Camera Norm: p1.rows  = " << n << std::endl;
// //     Mat tmp =  p1.col(0)-params.K.at<double>(0,2);
//     p1.col(0) = p1.col(0)-params.K.at<double>(0,2);
//     std::cout << "DB2.1.a.1 " << std::endl;
//     p1.col(1) = p1.col(1)-params.K.at<double>(1,2);
//        std::cout << "DB2.1.a.2 " << std::endl;
//     p1.col(0) = p1.col(0).mul(1.0/params.K.at<double>(0,0));
//        std::cout << "DB2.1.a.3 " << std::endl;
//     p1.col(1) = p1.col(1).mul(1.0/params.K.at<double>(1,1));
//     
//     //  p2
//     p2.col(0) = p2.col(0)-params.K.at<double>(0,2);
//     p2.col(1) = p2.col(1)-params.K.at<double>(1,2);
//     p2.col(0) = p2.col(0).mul(1.0/params.K.at<double>(0,0));
//     p2.col(1) = p2.col(1).mul(1.0/params.K.at<double>(1,1));
   
//     p1.col(0) = p1.col(0).mul(1.0/ params.K.at<double>(0,2));
//     p1.col(1) = p1.col(1).mul(1.0/ params.K.at<double>(1,2));
//     p1.col(0) = p1.col(0).mul(1.0/ params.K.at<double>(0,0));
//     p1.col(1) = p1.col(1).mul(1.0/ params.K.at<double>(1,0));
//     
//     //  p2
//     p2.col(0) = p2.col(1).mul(1.0/ params.K.at<double>(0,2));
//     p2.col(1) = p2.col(0).mul(1.0/ params.K.at<double>(1,2));
//     p2.col(0) = p2.col(1).mul(1.0/ params.K.at<double>(0,0));
//     p2.col(1) = p2.col(0).mul(1.0/ params.K.at<double>(1,0));
//     //  p1
//     p1.col(0) -= params.K.at<double>(0,2);
//     p1.col(1) -= params.K.at<double>(1,2);
//     p1.col(0) /= params.K.at<double>(0,0);
//     p1.col(1) /= params.K.at<double>(1,0);
//     
//     //  p2
//     p2.col(0) -= params.K.at<double>(0,2);
//     p2.col(1) -= params.K.at<double>(1,2);
//     p2.col(0) /= params.K.at<double>(0,0);
//     p2.col(1) /= params.K.at<double>(1,0);
    return;
}
