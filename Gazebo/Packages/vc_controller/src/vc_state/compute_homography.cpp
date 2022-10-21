#include "vc_state/img_tools.h" 
using namespace cv;
using namespace std;
int compute_homography(const Mat&img,
                       const vc_parameters & params, 
                       const vc_desired_configuration & desired_configuration,
                       vc_homograpy_matching_result& result) {
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
	result.p1.release();
	result.p2.release();
	result.p1 = Mat(goodMatches.size(),2,CV_64F);
	result.p2 = Mat(goodMatches.size(),2,CV_64F);
    cout << "n good matches " << goodMatches.size() << endl;
	for(int i = 0; i < goodMatches.size(); i++){
		//-- Get the keypoints from the good matches
// 		result.p1.push_back(Mat(desired_configuration.kp[goodMatches[i].queryIdx].pt));
// 		result.p2.push_back(Mat(kp[goodMatches[i].trainIdx].pt));
        
        int idx = goodMatches[i].queryIdx;
		
        Mat tmp = Mat(desired_configuration.kp[idx].pt).t();
        tmp.copyTo(result.p1.row(i));
        tmp.release();
        idx = goodMatches[i].trainIdx;
        tmp = Mat(kp[idx].pt).t();
		tmp.copyTo(result.p2.row(i));
	}

	//computing error
	
	
	result.mean_feature_error = norm(result.p1,result.p2)/(double)result.p1.rows;
	// Finding homography
    Mat a ,b;
    result.p1.convertTo(a,CV_32FC2);
    result.p2.convertTo(b,CV_32FC2);
	result.H = findHomography(a,b ,RANSAC, 0.5);
	if (result.H.rows==0)
		return -1;
	/************************************************************* Draw matches */
    
	result.img_matches = Mat::zeros(img.rows, img.cols * 2, img.type());
	drawMatches(desired_configuration.img, desired_configuration.kp, img, kp,
			goodMatches, result.img_matches,
			Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return 0;
}
