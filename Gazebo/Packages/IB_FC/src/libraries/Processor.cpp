#include "Processor.hpp"

/*
	function: constructor
*/
Processor::Processor(){}

/*
	function: destructor
*/
Processor::~Processor(){
	if(SEND_RECEIVE_SET){
		delete [] s;
		delete [] r;
	}
	if(DONE_SET) delete [] DONE;
}

/*
	function: takePicture
	description: sets the matrix with the image from the camera
	to process it. its an opencv mat obtained from the msg
*/
void Processor::takePicture(Mat img){
	this->img = img;
}

/*
	funcion: getImageDescription
	description: returns the computed image description ready to send as msg
*/
IB_FC::image_description Processor::getImageDescription(){
	return id;
}

/*
	function: getGM
	description: returns the geometric constraint corresponding to 
	the given neighbor
	params:	
		index: index of the agent we want to know about. This
		index depends of the index the agent appears in the array of neighbors s.
	returns:
		the geometric constraint related to both agents
		for example, if this drone is i=1, and neighbor j=2,
		returns Geometric constrain_{1,2}
*/ 
IB_FC::geometric_constraint Processor::getGM(int index){
	return gm[index];
}

/*
	function: BRCommunicationSends
	description: computes, in a brute force way, how the drone will communicate with the 
	neighbors
	params:
		label: label of the actual drone
		n_neigh: number of neighbors for this drone
		neighbors: ptr to the array with the neighbors for this drone
		n_s: ptr to save the the number of agents this drone will send information
*/
int *Processor::BRCommunicationSends(int label, int n_neigh, int *neighbors, int *n_s){	
	//count the neighbors that will send to this agent their information
	for (int i=0;i<n_neigh;i++)
		if(label < neighbors[i])
			ns++;

	//create array
	s = new int[ns];
	ns = 0;

	//fill the arrays	
	for (int i=0;i<n_neigh;i++)
		if(label < neighbors[i]){
			s[ns] = neighbors[i];
			ns++;

			IB_FC::geometric_constraint g;
			gm.push_back(g);
//             vector<Point2f> p1,p2;
//             pi.push_back(p1);
//             pj.push_back(p2);

			if(SHOW_MATCHING){
				vector<KeyPoint> k1,k2;
				vector<DMatch> m;
				matchesNeighbors.push_back(m);
				kp_j.push_back(k2);
				kp_i.push_back(k1);
				string j = to_string(neighbors[i]);
				string mk("mkdir "+output_dir+j+"/matching");
				system(mk.c_str());	
			}
		}
			
	*n_s = ns;
	return s;
}

/*
	function: BRCommunicationReceives
	description: computes, in a brute force way, how the drone will receive information
	params:
		label: label of the actual drone
		n_neigh: number of neighbors for this drone
		neighbors: ptr to the array with the neighbors for this drone
		n_s: ptr to save the the number of agents this drone will receive information
*/
int *Processor::BRCommunicationReceives(int label, int n_neigh, int *neighbors, int *n_r){
	for (int i=0;i<n_neigh;i++)
		if(label > neighbors[i]) 
			nr++;
		
	r = new int[nr];
	nr = 0;
 cout << "------------- BRC_Receives ------------- \n" << flush;
	for (int i=0;i<n_neigh;i++)
		if(label > neighbors[i]){			
//             vector<Point2f> p1,p2;
//             pi.push_back(p1);
//             pj.push_back(p2);
			r[nr] = neighbors[i];			
			nr++;
            
		}
		
     cout << "------------- BRC_Receives -------------" << n_neigh << " \n" << flush;

	*n_r = nr;
	return r;	
}

/*
	function: OptCommunicationSends
	description:
	params:
		g: Directed graph associated to the formation, can be obtained from getGraph function 
		label: label of the actual drone
		n_neigh: number of neighbors for this drone
		neighbors: ptr to the array with the neighbors for this drone
		n_s: ptr to save the the number of agents this drone will receive information
		
*/
int *Processor::OptCommunicationSends(DirectedGraph g, int label, int *n_s){	
	g.loadBalance();
	ns =  g.countConnections(label,g);
	s = new int[ns]; 
	g.getConnections(label,g,s);
	*n_s = ns;

	for(int i=0;i<ns;i++){
		IB_FC::geometric_constraint g;
		gm.push_back(g);
//         vector<Point2f> p1,p2;
//         pi.push_back(p1);
//         pj.push_back(p2);
        
		if(SHOW_MATCHING){
			vector<KeyPoint> k1,k2;
			vector<DMatch> m;
			matchesNeighbors.push_back(m);
			kp_j.push_back(k2);
			kp_i.push_back(k1);
			string j = to_string(s[i]);
			string mk("mkdir "+output_dir+j+"/matching");
			system(mk.c_str());	
		}
	}
			
	return s;
}

/*
	function: 
	description: 
	params:
		label: label of the actual drone
		n_neigh: number of neighbors for this drone
		neighbors: ptr to the array with the neighbors for this drone
		n_r: ptr to save the the number of agents this drone will receive information
*/
int *Processor::OptCommunicationReceives(int label, int n_neigh, int *neighbors, int *n_r){
	nr = 0;	
	int possible = 1;

	for (int i=0;i<n_neigh;i++){
		possible = 1;
		for(int j=0;j<ns;j++)		
			if(neighbors[i]==s[j]){
				possible = 0;
				break;
			}

		if(possible)
			nr++;
	}

	*n_r = nr;
	r = new int[nr];
	nr = 0;
	 cout << "------------- Opt_Receives ------------- \n" << flush;
	for (int i=0;i<n_neigh;i++){
//         vector<Point2f> p1,p2;
//             pi.push_back(p1);
//             pj.push_back(p2);
		possible = 1;
		for(int j=0;j<ns;j++)		
			if(neighbors[i]==s[j]){
				possible = 0;
				break;
			}

		if(possible){
			r[nr] = neighbors[i];			
			nr++;
		}
	}
cout << "------------- Opt_Receives -------------" << n_neigh << " \n" << flush;
	return r;	
}

/*
	Function: setProperties
	description: sets some properties needed for the processor
	params:
		label: label of this agent
		matching: if we want to show the matching between agetns
		controller_type: which controller we will use
	
*/
void Processor::setProperties(int label,int n_agents, int matching,int controller_type,string input_dir,string output_dir){
	this->label = label;
	this->SHOW_MATCHING = matching;
	this->controller_type = controller_type;
	this->input_dir = input_dir;
	this->output_dir = output_dir;
	if(!isHomographyConsensus(controller_type)){ 
		this->nlevels=4; this->patchSize = 15; //to improve matching
		DONE = new int[n_agents];//to verify if we have decompose with opencv the first time
		for(int i=0;i<n_agents;i++)
			DONE[i] = 0;
		DONE_SET = 1;
	}
	for (int i = 0; i < n_agents; i++)
    {
        vector<Point2f> p1,p2;
        pi.push_back(p1);
        pj.push_back(p2);
    }
	orb = ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,cv::ORB::HARRIS_SCORE,patchSize,fastThreshold);
	readK();
}

/*
	function: detectAndCompute
	description: process the image to obtain key points and descriptors. It also
	prepares the msg to send to neighbors
	params:
		pose: array with six elemenths indicating the pose
*/
void Processor::detectAndCompute(double *pose){
	/************************************************************ computing with ORB*/
	kp.clear();
	orb->detect(img, kp);
	orb->compute(img, kp, descriptors);

	/************************************************************ Prepare msg to publish descriptors*/
	int rows = descriptors.rows, cols = descriptors.cols;
	id.des.cols = cols; id.des.rows = rows; 
	id.des.data.clear();//clear previous data
	
	for (int i=0; i<rows; i++)
		for (int j=0; j<cols; j++)
		    id.des.data.push_back(descriptors.at<unsigned char>(i,j));

	/************************************************************ Prepare msg to publish kp*/
	id.kps.clear();
	for(std::vector<KeyPoint>::size_type i = 0; i != kp.size(); i++){
		IB_FC::key_point k;
		k.angle = kp[i].angle; k.class_id = kp[i].class_id;k.octave = kp[i].octave;
		k.pt[0] = kp[i].pt.x; k.pt[1] = kp[i].pt.y; 
		k.response = kp[i].response; k.size=kp[i].size;
		id.kps.push_back(k);		
	}
	//add aditional data and say that you have calculated everything
	id.j = label;

	for(int i=0;i<6;i++)
		id.pose[i] = pose[i];

	COMPUTED = 1;
}

/* 
	Function: getGeometricConstraint
	description: gets the key points and descriptors from neighbors (trough the msg) and computes the geometric constraint
	 	saving also the additional information from the msg.
	params: 
		msg: ptr to the message with the key points and descriptors from neighbor
		you: ptr to a variable where the label of the neighbor of the msg is stored
		pose_i: array to store the pose of this agent
		pose_j: array to store the pose of the neighbor
		SUCCESS: ptr to a variable that is a flag to indicate if the geometric constraint could 
			be computed
		n_matches: ptr to variable to store the good matches between both agents
		R: posible rotation relative rotation matrix (in a vector of size 9) obtained from
			opencv function recover pose. It is only given the first time and use for epipolar
			consensus
		t: posible relative translation vector (int a vector of size 3) obtained from 
			opencv function recover pose. It is only given the first time and used for epipolar
			consensus
*/
Mat Processor::getGeometricConstraint(const IB_FC::image_description::ConstPtr& msg, int *you, double *pose_i, double *pose_j, int *SUCCESS, int *n_matches, double *R, double *t){	
	//obtain information from thet msg
	int cols = msg->des.cols, rows = msg->des.rows;
	*you = msg->j; Mat null(3,3,0);
	int index = find(*you,s,ns); //index of this neighbor in array s

	//saving pose
	for(int i=0;i<6;i++)
		pose_j[i] = msg->pose[i];

	//fill the descriptors matrix
	Mat dn(rows,cols,0.0);
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++)		
			dn.at<unsigned char>(i,j) = msg->des.data[i*cols + j];

	//fill the kp vector
	vector<KeyPoint> kn;
	if(SHOW_MATCHING){ kp_j[index].clear(); kp_i[index].clear(); }
	for(std::vector<KeyPoint>::size_type i = 0; i != msg->kps.size(); i++){
		KeyPoint k;
		k.angle = msg->kps[i].angle; k.class_id = msg->kps[i].class_id;k.octave = msg->kps[i].octave;
		k.pt.x = msg->kps[i].pt[0]; k.pt.y = msg->kps[i].pt[1]; 
		k.response = msg->kps[i].response; k.size=msg->kps[i].size;
    		kn.push_back(k);
		if(SHOW_MATCHING==1){ kp_j[index].push_back(k);	kp_i[index].push_back(kp[i]); }
	}

	/************************************************************* Using flann for matching*/
	if( !COMPUTED || (rows == 0 && cols == 0)) return null; //if we dont have our own kp and descriptors

  	vector<vector<DMatch>> matches;
	FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
 	matcher.knnMatch(descriptors,dn,matches,2);
	vector<DMatch> goodMatches;

	/************************************************************* Processing to get only goodmatches*/	
	if(SHOW_MATCHING) matchesNeighbors[index].clear();

	for(int i = 0; i < matches.size(); ++i)
		if (matches[i][0].distance < matches[i][1].distance * RATIO){
			goodMatches.push_back(matches[i][0]);
			if(SHOW_MATCHING) matchesNeighbors[index].push_back(matches[i][0]);
		}   		

	*n_matches = goodMatches.size();

	/************************************************************* Finding geometric constraint */
	 //-- transforming goodmatches to points		
// 	vector<Point2f> p1; vector<Point2f> p2; 
// 	vector<Point2f> * p1 = pi.at(index);
//     p1.clear();
// 	vector<Point2f> * p2 = pj.at(index);
//     p2.clear();
	pi[*you].clear();
	pj[*you].clear();
	
	for(int i = 0; i < goodMatches.size(); i++){
		//-- Get the keypoints from the good matches
		pi[*you].push_back(kp[goodMatches[i].queryIdx].pt);
		pj[*you].push_back(kn[goodMatches[i].trainIdx].pt);
	}

	Mat GM;	
// 
// 	if(isHomographyConsensus(controller_type) && goodMatches.size() > 4){
// 		vector<int> mask;
// 		GM = findHomography(pj[index], pi[index] ,RANSAC, 1,mask);
// 		/*if(controller_type == 7){ //if we do not need to normalize homography
// 			vector<Point2f> fp, tp;
// 			for(int i=0;i<mask.size();i++){
// 				if(mask[i]){
// 					fp.push_back(pj[index][i]);
// 					tp.push_back(pi[index][i]);
// 				}
// 			}
// 			GM = H_from_points(tp,fp,1);
// 			double norm=sqrt(GM.at<double>(0,0)*GM.at<double>(0,0)+GM.at<double>(1,0)*GM.at<double>(1,0)+GM.at<double>(2,0)*GM.at<double>(2,0));
// 			double sign = 1;
// 			if(GM.at<double>(2,2)<0) sign = sign*-1;
// 			GM = sign*(1.0/norm)*GM;
// 
// 		}*/
// 
// 	}else if(goodMatches.size() > 12 && !epipoleConsensus(controller_type)){
// 		Mat mask;
// 		GM = findEssentialMat(pj[index], pi[index], K, RANSAC, 0.999, 1.0, mask); //compute essential matrix
// 		
// 		if(!DONE[*you]){
// 			Mat Rotation, translation;			
// 			recoverPose(GM, pj[index], pi[index], K, Rotation, translation, mask);//decompose using the mask
// 			for(int i=0;i<3;i++){
// 				t[i] = translation.at<double>(i); //for this agent process
// 				gm[index].t[i] = translation.at<double>(i);  //for the neighbor in the message
// 				for(int j=0;j<3;j++){
// 					R[i*3+j] = Rotation.at<double>(i,j); //for this agent process
// 					gm[index].R[i*3+j] = Rotation.at<double>(i,j); //for the neighbor in the message
// 				}
// 			}
// 			DONE[*you]  =1;
// 		}	
// 	}else if(goodMatches.size() > 12 && epipoleConsensus(controller_type)){
// 		Mat mask; double x_mid = K.at<double>(0,2),y_mid = K.at<double>(1,2); int nn = pi[index].size(); 
// 		for(int i=0;i<nn;i++){
// 			pi[index][i].x-=x_mid;
// 			pj[index][i].x-=x_mid;
// 			pi[index][i].y = y_mid -pi[index][i].y;
// 			pj[index][i].y = y_mid -pj[index][i].y;  
// 		}		
// 		GM = findFundamentalMat(pi[index],pj[index],FM_RANSAC,1.0,0.99,mask);
// 	}

	/*********************************************************** Save information in menssage object */
// 	gm[index].n_matches = goodMatches.size();
// 	gm[index].i = label;
// 	
// 	for(int i=0;i<6;i++)
// 		gm[index].pose[i] = pose_i[i];			
// 
// 	//fill the geoemtric constraint
// 	for(int i=0;i<3;i++)
// 		for(int j=0;j<3;j++)
// 			gm[index].constraint[i*3+j] = GM.at<double>(i,j);

	if(goodMatches.size() > 0)
		*SUCCESS = 1;

	return GM;
}

/*
	function: matchingCallback
	description: this function receives an image from the neighbor and, using the matches previously computed
		, saves and image showing the matching. This code will be executed if the SHOW_MATCHING is 1.
*/
void Processor::matchingCallback(const sensor_msgs::Image::ConstPtr& msg){	
	try{		
		//receive image
		Mat img_j = cv_bridge::toCvShare(msg,"bgr8")->image;
		//create empty images to draw keypoints
		Mat img_kp = Mat::zeros(img.rows, img.cols, img.type()); 
		Mat img_kp_j = Mat::zeros(img_j.rows, img_j.cols, img_j.type());
		//label of the neighbor agent
		int j = msg->header.frame_id[11]-'0';
		//find the index for this matching
		int index = find(j,s,ns);
		if(matchesNeighbors[index].size() == 0) return;

		/************************************************************* Draw matches */
		drawKeypoints(img, kp_i[index], img_kp,Scalar::all(-1));
		drawKeypoints(img_j, kp_j[index], img_kp_j,Scalar::all(-1));

		Mat img_matches = Mat::zeros(img.rows, img.cols * 2, img.type());
		drawMatches(img_kp, kp_i[index], img_kp_j, kp_j[index], 
					matchesNeighbors[index], img_matches, 
					Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		string name = to_string(j)+"/matching/"+to_string(count)+".png";		
		imwrite(output_dir+name, img_matches);
		count ++;

	}catch (cv_bridge::Exception& e){
	 	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
   }
	
}

/*
	function: readK
	description: reads the calibration matrix K from a file of the given input_dir
*/
void Processor::readK(){
	fstream inFile;
	string file("K.txt");

	double x=0; int i=0,j=0;
	//open file
	inFile.open(input_dir+file);
	//read file
	while (inFile >> x) {
		K.at<double>(i,j) = x;
		j++; if(j==3){i++;j=0;}
	}

	//close file
	inFile.close();
}

/* 
	function: H_from_points
	description: Find homography H, such that fp is mapped to tp using the 
		linear DLT method. Points are conditioned automatically. 
		It is recommended to use the ransac funcion from opencv
		and then use the inlier points in this function
	params:
		fp: points image 1
		tp: points image 2
		normalized: if we want to normalized the homography by the 3x3 matrix element		
	returns:
		H: Homography
*/
Mat Processor::H_from_points(vector<Point2f> &fp,vector<Point2f> &tp, int normalized){
	
	if(fp.size() != tp.size()) cout << "number of points do not match" << endl;

	//condition points (important for numerical reasons)
	// we condition de from_points (fp) and the to points (tp)
	//-----------------------------------------get mean for  x and y
	double fp_mean_x = 0, fp_mean_y = 0, n = fp.size(); //to save the mean
	double tp_mean_x = 0, tp_mean_y = 0; //to save the mean
	int n_matches = tp.size(); 

	for(int i=0;i<n_matches;i++){
		fp_mean_x+=fp[i].x; fp_mean_y+=fp[i].y;
		tp_mean_x+=tp[i].x; tp_mean_y+=tp[i].y;		
	}
	fp_mean_x = fp_mean_x/n +1e-9; fp_mean_y=fp_mean_y/n+1e-9;
	tp_mean_x = tp_mean_x/n +1e-9; tp_mean_y=tp_mean_y/n+1e-9;
	
	//------------------------------------------get standard deviation for x and y
	double fp_std_x = 0, fp_std_y = 0;
	double tp_std_x = 0, tp_std_y = 0;
	
	for(int i=0;i<n_matches;i++){
		fp_std_x+=(fp[i].x-fp_mean_x)*(fp[i].x-fp_mean_x); fp_std_y+=(fp[i].y-fp_mean_y)*(fp[i].y-fp_mean_y);
		tp_std_x+=(tp[i].x-tp_mean_x)*(tp[i].x-tp_mean_x); fp_std_y+=(tp[i].y-tp_mean_y)*(tp[i].y-tp_mean_y);
	}

	double fp_maxstd = max(fp_std_x,fp_std_y) + 1e-9;
	double tp_maxstd = max(tp_std_x,tp_std_y) + 1e-9;
	
	//--------------------------------------construct needed matrix
	Mat C1 = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	C1.at<double>(0,0) = 1.0/fp_maxstd;
	C1.at<double>(1,1) = 1.0/fp_maxstd;
	C1.at<double>(0,2) = -fp_mean_x/fp_maxstd;
	C1.at<double>(1,2) = -fp_mean_y/fp_maxstd;

	Mat C2 = (Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	C2.at<double>(0,0) = 1.0/tp_maxstd;
	C2.at<double>(1,1) = 1.0/tp_maxstd;
	C2.at<double>(0,2) = -tp_mean_x/tp_maxstd;
	C2.at<double>(1,2) = -tp_mean_y/tp_maxstd;

	double a[n_matches*2][9];
	for(int i=0;i<n_matches;i++){		
		Mat f_point = (Mat_<double>(3,1) << fp[i].x,fp[i].y, 1.0);
		Mat t_point = (Mat_<double>(3,1) << tp[i].x,tp[i].y, 1.0);
		
		Mat fp_i = C1*f_point; Point2f f; f.x = fp_i.at<double>(0); f.y = fp_i.at<double>(1);
		Mat tp_i = C2*t_point; Point2f t; t.x = tp_i.at<double>(0); t.y = tp_i.at<double>(1);

		double a_2i[9] = {-f.x,-f.y,-1.0,0.0,0.0,0.0,t.x*f.x,t.x*f.y,t.x};
		double a_2i1[9] = {0.0,0.0,0.0,-f.x,-f.y,-1.0,t.y*f.x,t.y*f.y,t.y};
		for(int j=0;j<9;j++){
			a[2*i][j] = a_2i[j];
			a[2*i+1][j] = a_2i1[j];
		}
	}

	Mat A(n_matches*2,9,CV_64F,a);

	Mat W,U,Vt;
	SVD::compute(A,W,U,Vt);	

	double Hom[9];
	for(int i=0;i<9;i++)
		Hom[i] = Vt.at<double>(8,i);

   	Mat H = Mat(3, 3, CV_64F, Hom);
	H = C2.inv()*H*C1;

	if(normalized) return H/H.at<double>(2,2);

	return H;
}
