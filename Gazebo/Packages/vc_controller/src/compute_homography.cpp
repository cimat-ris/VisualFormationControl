#include "vc_controller/img_tools.h" 

using namespace vcc;

int vcc::compute_homography(
    const cv::Mat & img,
    const parameters & params, 
    const desired_configuration & Desired_Configuration,
    homograpy_matching_result & result) {
	
    /* keypoints and descriptors */
	cv::Mat descriptors; 
    std::vector<cv::KeyPoint> kp; 
  
	/*** Creatring ORB object ***/
	cv::Ptr<cv::ORB> orb = cv::ORB::create(
        params.nfeatures,
        params.scaleFactor,
        params.nlevels,
        params.edgeThreshold,
        params.firstLevel,
        params.WTA_K,
        params.scoreType,
        params.patchSize,
        params.fastThreshold);
    
	orb->detect(img, kp);
	if (kp.size()==0)
		return -1;
	orb->compute(img, kp, descriptors);
    
    
	/******************** Using flann for matching *****************/
	cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
	std::vector<std::vector<cv::DMatch>> matches;
	matcher.knnMatch(
        Desired_Configuration.descriptors,
        descriptors,matches,2);

	/************** Processing to get only goodmatches ***********/
    
	std::vector<cv::DMatch> goodMatches;
	for(int i = 0; i < matches.size(); ++i) {
		if (matches[i][0].distance < matches[i][1].distance * params.flann_ratio)
				goodMatches.push_back(matches[i][0]);
		}
	if (goodMatches.size()==0)
		return -1;

	/******************************* Findig homography **************/
    
	//-- transforming goodmatches to points
	result.p1.release();
	result.p2.release();
	result.p1 = cv::Mat(goodMatches.size(),2,CV_64F);
	result.p2 = cv::Mat(goodMatches.size(),2,CV_64F);
    
    for(int i = 0; i < goodMatches.size(); i++){
		//-- Get the keypoints from the good matches
        
        int idx = goodMatches[i].queryIdx;
		
        cv::Mat tmp = cv::Mat(Desired_Configuration.kp[idx].pt).t();
        tmp.copyTo(result.p1.row(i));
        tmp.release();
        idx = goodMatches[i].trainIdx;
        tmp = cv::Mat(kp[idx].pt).t();
		tmp.copyTo(result.p2.row(i));
	}

	//computing error
	result.mean_feature_error = norm(result.p1,result.p2)/(double)result.p1.rows;
	// Finding homography
    cv::Mat a ,b;
    result.p1.convertTo(a,CV_32FC2);
    result.p2.convertTo(b,CV_32FC2);
	result.H = cv::findHomography(a,b ,cv::RANSAC, 0.5);
	if (result.H.rows==0)
		return -1;
	/************************* Draw matches ******************/
    
	result.img_matches = cv::Mat::zeros(img.rows, img.cols * 2, img.type());
	cv::drawMatches(
        Desired_Configuration.img,
        Desired_Configuration.kp, 
        img, kp, goodMatches, 
        result.img_matches,
        cv::Scalar::all(-1), 
        cv::Scalar::all(-1), 
        std::vector<char>(), 
        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return 0;
}
