#include "utils_img.h"

using namespace cv;
using namespace std;

vc_homograpy_matching_result::vc_homograpy_matching_result(): mean_feature_error(1e10) {}

int compute_homography(const Mat&img, const vc_parameters&params, const vc_desired_configuration&desired_configuration, vc_homograpy_matching_result& result) {
	/*** KP ***/
	Mat descriptors; vector<KeyPoint> kp; // kp and descriptors for current image

	/*** Creatring ORB object ***/
	Ptr<ORB> orb = ORB::create(params.nfeatures,params.scaleFactor,params.nlevels,params.edgeThreshold,params.firstLevel,params.WTA_K,params.scoreType,params.patchSize,params.fastThreshold);
	orb->detect(img, kp);
	if (kp.size()==0)
		return -1;
	orb->compute(img, kp, descriptors);
	/************************************************************* Using flann for matching*/
	FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
	vector<vector<DMatch>> matches;
	matcher.knnMatch(desired_configuration.descriptors,descriptors,matches,2);

	/************************************************************* Processing to get only goodmatches*/
	vector<DMatch> goodMatches;
	for(int i = 0; i < matches.size(); ++i) {
		if (matches[i][0].distance < matches[i][1].distance * params.flann_ratio)
				goodMatches.push_back(matches[i][0]);
		}
	if (goodMatches.size()==0)
		return -1;

	/************************************************************* Findig homography */
	//-- transforming goodmatches to points
	result.p1.clear();
	result.p2.clear();
	for(int i = 0; i < goodMatches.size(); i++){
		//-- Get the keypoints from the good matches
		result.p1.push_back(desired_configuration.kp[goodMatches[i].queryIdx].pt);
		result.p2.push_back(kp[goodMatches[i].trainIdx].pt);
	}

	//computing error
	Mat a = Mat(result.p1); Mat b = Mat(result.p2);
	result.mean_feature_error = norm(a,b)/(float)result.p1.size();
	// Finding homography
	result.H = findHomography(result.p1, result.p2 ,RANSAC, 0.5);
	if (result.H.rows==0)
		return -1;
	/************************************************************* Draw matches */
	result.img_matches = Mat::zeros(img.rows, img.cols * 2, img.type());
	drawMatches(desired_configuration.img, desired_configuration.kp, img, kp,
			goodMatches, result.img_matches,
			Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return 0;
}

int select_decomposition(const vector<Mat> &Rs,const vector<Mat> &Ts,const vector<Mat> &Ns,
												const vc_homograpy_matching_result& matching_result,
												int &selected,Mat&Rbest,Mat &tbest) {
		if(selected == 0) {
			// To store the matrix Rotation and translation that best fix
			// Constructing the extrinsic parameters matrix for the actual image
			Mat P2 = Mat::eye(3, 4, CV_64F);

			double th = 0.1, nz = 1.0; //max value for z in the normal plane
			// Preparing the points for the test
			vector<Point2f> pp1; vector<Point2f> pp2;
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
						selected = 1;
					}
				}
			}
			// Process again, it is probably only in z axiw rotation, and we want the one with the highest nz component
			if (selected == 0){
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
						selected = 1;
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
