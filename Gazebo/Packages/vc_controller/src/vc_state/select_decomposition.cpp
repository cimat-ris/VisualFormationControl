#include "vc_state/img_tools.h" 
using namespace cv;
using namespace std;
int select_decomposition(const vector<Mat> &Rs,
                         const vector<Mat> &Ts,
                         const vector<Mat> &Ns,
                         const vc_homograpy_matching_result& matching_result,
                         bool &selected,
                         Mat&Rbest,Mat &tbest) {
		if(selected == false) {
			// To store the matrix Rotation and translation that best fix
			// Constructing the extrinsic parameters matrix for the actual image
			Mat P2 = Mat::eye(3, 4, CV_64F);

			double th = 0.1, nz = 1.0; //max value for z in the normal plane
			// Preparing the points for the test
			vector<Point2f> pp1; vector<Point2f> pp2;
//             Mat pp1; Mat pp2;
			pp1.push_back(matching_result.p1[0]);
			pp2.push_back(matching_result.p2[0]);

			// For every rotation matrix
			for(int i=0;i<Rs.size();i++){
				// Constructing the extrinsic parameters matrix for the desired image
				Mat P1; hconcat(Rs[i],Ts[i],P1);
				// To store the result
				Mat p3D;
				triangulatePoints(P1,P2,pp1,pp2,p3D); //obtaining 3D point
				// Transforming to homogeneus
				Mat point(4,1,CV_64F);
				point.at<double>(0,0) = p3D.at<float>(0,0) /p3D.at<float>(3,0);
				point.at<double>(1,0) = p3D.at<float>(1,0) /p3D.at<float>(3,0);
				point.at<double>(2,0) = p3D.at<float>(2,0) /p3D.at<float>(3,0);
				point.at<double>(3,0) = p3D.at<float>(3,0) /p3D.at<float>(3,0);
				// Verify if the point is in front of the camera. Also if is similar to [0 0 1] o [0 0 -1]
				// Giving preference to the first
				if(point.at<double>(2,0) >= 0.0 && fabs(fabs(Ns[i].at<double>(2,0))-1.0) < th ){
					if(nz > 0){
						Rs[i].copyTo(Rbest);
						Ts[i].copyTo(tbest);
						nz = Ns[i].at<double>(2,0);
						selected = true;
					}
				}
			}
			// Process again, it is probably only in z axiw rotation, and we want the one with the highest nz component
			
            if (selected == false){
				double max = -1;
				for(int i=0;i<Rs.size();i++){
					// Constructing the extrinsic parameters matrix for the desired image
					Mat P1; hconcat(Rs[i],Ts[i],P1);
					//to store the result
					Mat p3D;
					triangulatePoints(P1,P2,pp1,pp2,p3D); //obtaining 3D point
					// Transforming to homogeneus
					Mat point(4,1,CV_64F);
					point.at<double>(0,0) = p3D.at<float>(0,0) /p3D.at<float>(3,0);
					point.at<double>(1,0) = p3D.at<float>(1,0) /p3D.at<float>(3,0);
					point.at<double>(2,0) = p3D.at<float>(2,0) /p3D.at<float>(3,0);
					point.at<double>(3,0) = p3D.at<float>(3,0) /p3D.at<float>(3,0);

					if(point.at<double>(2,0) >= 0.0 && fabs(Ns[i].at<double>(2,0)) > max){
						Rs[i].copyTo(Rbest);
						Ts[i].copyTo(tbest);
						max = fabs(Ns[i].at<double>(2,0));
						selected = true;
					}
				}
			}
			//if not of them has been selected
			//now, we are not going to do everything again
		} else {//if we already selected one, select the closest to that one
            
			double min_t = 1e8, min_r = 1e8;
			Mat t_best_for_now, r_best_for_now;
			//choose the closest to the previous one
			for(int i=0;i<Rs.size();i++){
				double norm_diff_rot = norm(Rs[i],Rbest);
				double norm_diff_t = norm(Ts[i],tbest);
				if(norm_diff_rot < min_r){ Rs[i].copyTo(r_best_for_now); min_r=norm_diff_rot; }
				if(norm_diff_t < min_t){ Ts[i].copyTo(t_best_for_now); min_t=norm_diff_t; }
			}
			
			//save the best but dont modify it yet
			r_best_for_now.copyTo(Rbest);
			t_best_for_now.copyTo(tbest);
		}
		return 0;
}
 
