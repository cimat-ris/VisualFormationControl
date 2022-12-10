#include "Controller.hpp"

/*
	function: constructor
*/
Controller::Controller(){}

/*
	function: destructor
	params: before destroying everything, it writes import data
*/
Controller::~Controller(){
	if(ERRORS_SET) {delete [] errors_t; delete [] errors_psi;}
	if(FILTERS_SET){
		freeMatrix((char**)DONE,n_agents);
		freeMatrix((char**)xf,n_agents);
		freeMatrix((char**)yf,n_agents);
		freeMatrix((char**)zf,n_agents);
		freeMatrix((char**)yawf,n_agents);
	}
	if(GAMMA_SET) delete [] gamma;
}

/*
	function: setDesiredRelativePoses
	description: set the desired relative poses, this poses are calculated
	by formation.initRelativePoses
	params:
		desired relative poses given by every coordinate and yaw.
		These are give as a matrix in order to use it as 
		x_aster[i][j], representing the x coordinate of the relative translation
		vector between agent i and j.
*/
void Controller::setDesiredRelativePoses(double **x_aster,double **y_aster, double **z_aster, double **yaw_aster){
	this->x_aster = x_aster;
	this->y_aster = y_aster;
	this->z_aster = z_aster;
	this->yaw_aster = yaw_aster;
}

/*
	function: setProperties
	description: set some important propierties to define this controller
	and inits most of the arrays needed
	params:
		label: if of the drone using this controller
		n_agents: number of agents
		n_neigh: number of neighbors
		controller_type: controller type code
		Kv: gain for linear velocities
		Kw: gain for angular velocities
		A: Doubly stochastic matrix for altitude consensus
		input_dir: directory of input files
		output_dir: directory to save results
		gamma_info: flag to know if we have gamma_0 from file
*/
void Controller::setProperties(int label, int n_agents, int n_neigh, int controller_type, double Kv, double Kw, double **A, string input_dir, string output_dir, int gamma_file){
	//setting properties to object
	this->label = label;
	this->n_agents = n_agents;
	this->n_neigh = n_neigh;
	this->controller_type = controller_type;
	this->Kv = Kv; 
	this->Kw = Kw; 	
	this->input_dir = input_dir;
	this->output_dir = output_dir;

	//creating memory	
  	errors_psi = new double[n_agents];
	errors_t = new double[n_agents];
    velContributions = new bool[n_agents];
    for (int i = 0; i < n_agents ; i++)
        velContributions[i] = false;
	ERRORS_SET = 1;
	if(needsFilter(controller_type)){
		DONE =  (int **) createMatrix(sizeof(int), sizeof(int *), n_agents,n_agents);
		xf =  (double **) createMatrix(sizeof(double), sizeof(double *), n_agents,n_agents);
		yf =  (double **) createMatrix(sizeof(double), sizeof(double *), n_agents,n_agents);
		zf =  (double **) createMatrix(sizeof(double), sizeof(double *), n_agents,n_agents);
		yawf =  (double **) createMatrix(sizeof(double), sizeof(double *), n_agents,n_agents);
		for (int i=0;i<n_agents;i++)
			for(int j=0;j<n_agents;j++){
				DONE[i][j] = 0;
				xf[i][j]=0;yf[i][j]=0;zf[i][j]= 0.0;yawf[i][j]=0; 
			}
		FILTERS_SET = 1;
	}

	//initializing vectors
	for (int i=0;i<n_neigh;i++){
		errors_t[i] = -1;
		errors_psi[i] = -1;
	}

	//if altitude consensus is needed
	if(needsAltitudeConsensus(controller_type)){ 
		this->A = A; 
		gamma = new double[n_agents]; 
		GAMMA_SET = 1;
		if(gamma_file){//if we are given a file with gamma_0
			fstream inFile;
			string file("gamma.txt");
			int i=0; double x;
			//open file
			inFile.open(input_dir+file);
			//read file
			while (inFile >> x) {
				gamma[i] = x;
				i++;
			}
	
			//close file
			inFile.close();
			
		}else	
			for(int i=0;i<n_agents;i++) 
				gamma[i] = -1;
	}

	//if we are working with bearings
	if(bearingsNeeded(controller_type)){
		for(int i=0;i<n_agents;i++)
			for(int j=0;j<n_agents;j++){
				double norm = sqrt(x_aster[i][j]*x_aster[i][j]+y_aster[i][j]*y_aster[i][j]+z_aster[i][j]*z_aster[i][j]);
				if(norm!=0){
					x_aster[i][j]= x_aster[i][j]/norm;
					y_aster[i][j]= y_aster[i][j]/norm;
					z_aster[i][j]= z_aster[i][j]/norm;
				}
			}
	}

	readK();
}

/*
	function: resetVelocities
	description: turns velocities to 0 and computes information about the errors.
*/
void Controller::resetVelocities(){
	Vx = 0.0; Vy = 0.0; Vz = 0.0; Wz = 0.0;
    for(int i = 0; i < n_agents; i++)
        velContributions[i] = false;
}

/*
	function: getVelocities
	description: assign computed velocities in the given variables ptr
*/
void Controller::getVelocities(double *Vx, double *Vy, double *Vz, double *Wz){
	*Vx = this->Vx; *Vy = this->Vy; *Vz = this->Vz; *Wz = this->Wz;
}

/*
	function: setGamma
	description: sets the value of gamma given the agent label
	params:
		label: label of the agent we receive the value
		gamma: value of gamma for the given agent
*/
void Controller::setGamma(int label, double val){
	gamma[label] = val;
}

/*
	function: compute
	description: computes the velocities with the given controller type
	params:
		matches: amount of matches used to obtain the geometric constraint
		j: label of the neighbor
		GC: computed geometric constraint
		pose_i: pose obtained from sensors
		pose_j: pose from neighbor obtained from sensor
		R: posible rotation relative rotation matrix (in a vector of size 9) obtained from
			opencv function recover pose. It is only given the first time and use for epipolar
			consensus
		t: posible relative translation vector (int a vector of size 3) obtained from 
			opencv function recover pose. It is only given the first time and used for epipolar
			consensus
*/
// void Controller::compute(int matches, int j, Mat &GC,double *pose_i, double *pose_j,double *R, double *t, vector<Point2f> kp_j,vector<Point2f> kp_i){
// 	this->pose_i = pose_i; this->pose_j = pose_j; 
// cout << "------------- DB3 ------------- \n";
// 	switch(controller_type){
// 		case 1: PBFCHD(matches,label,j,GC); break;
// 		case 2: PBFCED(matches,label,j,GC,R,t); break;
// 		case 3: PBFCHDSA(matches,label, j, GC); break;
// 		case 4: PBFCEDSA(matches,label,j,GC,R,t); break;
// 		case 5: RBFCHD(matches,label,j,GC); break;
// 		case 6: RBFCED(matches,label,j,GC,R,t); break;
// 		case 7: IBFCH(matches,label,j,GC); break;
// 		case 8: IBFCE(matches,label,j,GC,R,t); break;
// 		case 9: EBFC(matches, label, j, GC); break;
//         case 10: IBFCF(matches, label, j, kp_j, kp_i); break; 
// 		default: cout << "Computing nothing..."<<endl; 
// 	}
// }


bool Controller::incompleteComputedVelocities(){
    int count = 0;
    for (int i = 0 ; i < n_agents; i++)
    {
        if(velContributions[i] == true)
            count++;
    }
    
    if (count >= n_neigh)
        return false;
    
    return true;
    
}
    
    
/*
	function: haveComputedVelocities
	description: checks if has computed the velocities with all its neighbors
	params: 
		et: pointer to the double variable to save the information about the translation error
		epsi: pointer to the double variable to save the information about the rotation error
	returns:
		0: false
		1: true
*/
int Controller::haveComputedVelocities(double *et, double *epsi){
	//to sum the errors and verify if we calculated with every neighbor
	double count=0.0,err_t = 0,err_psi = 0;
	for(int i=0;i<n_agents;i++){
		if(errors_psi[i]!=-1){
			err_t+=errors_t[i];
			err_psi+=errors_psi[i];
			count+=1.0;
// 			errors_t[i] = -1;
// 			errors_psi[i] = -1; 
		}
	}
	//Verify if we computed everything		
	if(n_neigh <= count){ 
        cout << "---- " << label << " Controller : have computed velocities " << count << endl << flush;
		//save velocities
		double data[4] = {Vx,Vy,Vz,Wz};
		appendToFile(output_dir+"/velocities.txt",data,4);
		//save errors
		err_t/=(double)count; err_psi/=(double)count;		
		*et = err_t; *epsi = err_psi;
		data[0] = err_t; data[1] = err_psi;
		appendToFile(output_dir+"/errors.txt",data,2);
        
        for(int i = 0; i < n_agents ; i++){
            errors_t[i] = -1;
			errors_psi[i] = -1; 
        }
		if(needsAltitudeConsensus(controller_type)){
			//make a step in scale
			data[0] = gamma[label];
			appendToFile(output_dir+"/gamma.txt",data,1);
			Mat Am = Mat(n_agents, n_agents, CV_64F);
			Mat gam = Mat(n_agents, 1, CV_64F);
			for(int i=0;i<n_agents;i++){
				gam.at<double>(i,0) = gamma[i];
				for(int j=0;j<n_agents;j++)
					Am.at<double>(i,j) = A[i][j];
			}
			Mat next = Am*gam;	
			for(int i=0;i<n_agents;i++)			
				gamma[i] = next.at<double>(i,0);
		}
		return 1;
	}	
cout << "---- " << label << " Controller : have NOT computed velocities " << count << endl << flush;
	return 0;
}

/*
	function: gammaInitialized
	description: verifies if the gamma_0 vector is ready to star
		computing its consensus. If the altitude consensus is needed
	returns:
		1: ready
		0: not ready
*/
int Controller::gammaInitialized(){
	if(!needsAltitudeConsensus(controller_type))//gamma not needed
		return 1;
	
	int c = 0;
	for(int i=0;i<n_agents;i++)	
		if(gamma[i]!=-1) c++;

	if(c==n_agents)
		return 1;

	return 0;
}

/*
	function: PBFCHD (controller_type: 1)
	description: Position Based Formation Control using Homography Decomposition
	params:
		matches: amount of matches used to obtain the homography
		i: label of this agent
		j: label of the neighbor
		H: homography
*/
void Controller::PBFCHD(int matches,int i, int j, Mat &H){
	vector<Mat> rotations;
	vector<Mat> translations;
	vector<Mat> normals;
	decomposeHomographyMat(H, K, rotations,translations, normals);

	if(!DONE[i][j]){
		DONE[i][j] = choose_first_homography_decomposition(matches, i,j,translations, rotations, normals);
		if(!DONE[i][j]) return;
	}
	
	double p_ij[3], yaw_ij;

	filter_pose(i,j,rotations,translations, p_ij, &yaw_ij);	
	yaw_ij = -yaw_ij; p_ij[0] = -p_ij[0]; p_ij[1] = -p_ij[1]; p_ij[2] = -p_ij[2];
	double norm = 0; for(int k=0;k<3;k++) norm+=(p_ij[k]*p_ij[k]);
	norm = sqrt(norm); p_ij[0]/=norm;p_ij[1]/=norm;p_ij[2]/=norm;
	/*
	Mat R_i = rotationZ(pose_i[5]); Mat R_j = rotationZ(pose_j[5]); Mat R_ij = R_i.t()*R_j;		
	Vec3d p_i(pose_i[0],pose_i[1],pose_i[2]); Vec3d p_j(pose_j[0],pose_j[1],pose_j[2]);
	Vec3f angles = rotationMatrixToEulerAngles(R_ij); yaw_ij = angles[2];
	Mat p_ij1 = R_i.t()*Mat(p_j-p_i); 
	double xij = p_ij1.at<double>(0,0);
	double yij = p_ij1.at<double>(1,0);
	double zij = p_ij1.at<double>(2,0);

	norm = sqrt(xij*xij+yij*yij+zij*zij); xij/=norm;yij/=norm;zij/=norm;
	p_ij[0] = xij; p_ij[1] = yij; p_ij[2]=zij;	*/

	Vx += Kv*(p_ij[0]-x_aster[i][j]);
	Vy += Kv*(p_ij[1]-y_aster[i][j]);	
	Vz += Kv*(p_ij[2]-z_aster[i][j]);	
	Wz += Kw*(yaw_ij-yaw_aster[i][j]);

	getError(matches, i,j,p_ij[0],p_ij[1],p_ij[2],yaw_ij);
}

/*
	function: PBFCED (controller_type: 2)
	description: Position Based Formation Control using Essential Matrix Decomposition
	params:
		matches: amount of matches used to obtain the homography
		i: label of this agent
		j: label of the neighbor
		E: Essential matrix
		R: posible rotation relative rotation matrix (in a vector of size 9) obtained from
			opencv function recover pose. It is only given the first time and use for epipolar
			consensus
		t: posible relative translation vector (int a vector of size 3) obtained from 
			opencv function recover pose. It is only given the first time and used for epipolar
			consensus
*/
void Controller::PBFCED(int matches,int i, int j, Mat &E,double *R, double *t){
	//to store the possible results
	double p_ij[3],yaw_ij;

	if(!DONE[i][j]){//use the first decomposition and choice to set the filters
		if(!matches) return;//not possible
		Mat Rot = Mat(3, 3, CV_64F, R);
		Vec3d angles = rotationMatrixToEulerAngles(Rot);
		yaw_ij = angles[2];
		//remember the position is already normalized
		p_ij[0] = t[1]; p_ij[1] = t[0]; p_ij[2] = t[2];
		yawf[i][j] = yaw_ij;
		xf[i][j] = p_ij[0]; yf[i][j] = p_ij[1];zf[i][j] = p_ij[2];
		DONE[i][j] = 1;
	}else{//decompose essential		
		Mat R1,R2,tr;
		decomposeEssentialMat(E,R1,R2,tr);		
		filter_pose_essential(i,j,R1,R2,tr, p_ij, &yaw_ij);	
	}
	 
	p_ij[0] = -p_ij[0]; p_ij[1] = -p_ij[1]; p_ij[2] = -p_ij[2];
	yaw_ij = -yaw_ij;
	
	Vx += Kv*(p_ij[0]-x_aster[i][j]);
	Vy += Kv*(p_ij[1]-y_aster[i][j]);	
	Vz += Kv*(p_ij[2]-z_aster[i][j]);	
	Wz += Kw*(yaw_ij-yaw_aster[i][j]);

	getError(matches, i,j,p_ij[0],p_ij[1],p_ij[2],yaw_ij);
}

/*
	function: PBFCHDSA (controller_type: 3)
	description: Position Based Formation Control using Homography Decomposition (Scale Aware)
	params:
		matches: amount of matches used to obtain the homography
		i: label of this agent
		j: label of the neighbor
		H: Homography
*/
void Controller::PBFCHDSA(int matches,int i, int j, Mat &H){
	vector<Mat> rotations;
	vector<Mat> translations;
	vector<Mat> normals;
	decomposeHomographyMat(H, K, rotations,translations, normals);

	if(!DONE[i][j]){
		DONE[i][j] = choose_first_homography_decomposition(matches, i,j,translations, rotations, normals);
		if(!DONE[i][j]) return;
	}
	
	double p_ij[3], yaw_ij;

	filter_pose(i,j,rotations,translations, p_ij, &yaw_ij);	
	yaw_ij = -yaw_ij; p_ij[0] = -p_ij[0]*gamma[i]; p_ij[1] = -p_ij[1]*gamma[i]; p_ij[2] = -p_ij[2]*gamma[i];

	Vx += Kv*(p_ij[0]-x_aster[i][j]);
	Vy += Kv*(p_ij[1]-y_aster[i][j]);	
	Vz += Kv*(p_ij[2]-z_aster[i][j]);	
	Wz += Kw*(yaw_ij-yaw_aster[i][j]);
	
	getError(matches, i,j,p_ij[0],p_ij[1],p_ij[2],yaw_ij);
}

/*
	function: PBFCEDSA (controller_type: 4)
	description: Position Based Formation Control using Essential Decomposition (Scale Aware)
	params:
		matches: amount of matches used to obtain the homography
		i: label of this agent
		j: label of the neighbor
		E: Essential
		R: posible rotation relative rotation matrix (in a vector of size 9) obtained from
			opencv function recover pose. It is only given the first time and use for epipolar
			consensus
		t: posible relative translation vector (int a vector of size 3) obtained from 
			opencv function recover pose. It is only given the first time and used for epipolar
			consensus
*/
void Controller::PBFCEDSA(int matches,int i, int j, Mat &E,double *R, double *t){
}

/*
	function: RBFCHD (controller_type: 5)
	description: Rigidity Based Formation Control using Homography Decomposition
	params:
		matches: amount of matches used to obtain the homography
		i: label of this agent
		j: label of the neighbor
		H: homography
*/
void Controller::RBFCHD(int matches,int i, int j, Mat &H){
}

/*
	function: RBFCED (controller_type: 6)
	description: Rigidity Based Formation Control using Homography Decomposition
	params:
		matches: amount of matches used to obtain the homography
		i: label of this agent
		j: label of the neighbor
		E: Essential matrix
		R: vector of 1x9 with information about the possible rotation matrix the
			first decomposition
		t: vector of 1x3 with infotmation about the possible translation vector for the
			first decomposition
*/
void Controller::RBFCED(int matches,int i, int j, Mat &E, double *R, double *t){
}

/*
	function: IBFCH (controller_type: 7)
	description: Image-Based Formation Control using Homography. 
		this controller is the implementation from Montijano et 
		al paper 'Vision-based Distributed Formation Control
		without an External Positioning System'. CODE: 7
	params:
		matches: number of matches used to obtain the homography
		i: label of the agent whose reference frame was used to 
			obtain the homography (this agent)
		j: label of the agent used to compute the homography
		H: Homography as opencv Mat
*/
void Controller::IBFCH(int matches, int i, int j, Mat &H){
	//create rotation matrices to obtain the rectified matrix
	Mat RXj = rotationX(pose_j[3]); Mat RXi = rotationX(pose_i[3]);
	Mat RYj = rotationY(pose_j[4]); Mat RYi = rotationY(pose_i[4]);
	//rectification to agent i and agent j
	Mat Hir = RXi.inv() * RYi.inv()*K.inv();
	Mat Hjr = RXj.inv() * RYj.inv()*K.inv();
	//rectified matrix
	Mat Hr = Hir*H*Hjr.inv();

	double p_ij[3] = {-gamma[i]*Hr.at<double>(1,2),
                        -gamma[i]*Hr.at<double>(0,2),
                        1.0-Hr.at<double>(2,2)};
	double yaw_ij = -atan2(Hr.at<double>(1,0),Hr.at<double>(0,0));

	Vx += Kv*(p_ij[0]-x_aster[i][j]);
	Vy += Kv*(p_ij[1]-y_aster[i][j]);		
	Vz += Kv*(p_ij[2]-z_aster[i][j]);	
	Wz += Kw*(yaw_ij-yaw_aster[i][j]);	

	getError(matches,i,j,p_ij[0],p_ij[1],p_ij[2],yaw_ij);
}

/*
	function: IBFCE (controller_type: 8)
	description: Image-Based Formation Control using Essential Matrix.
	params:
		matches: number of matches used to obtain the homography
		i: label of the agent whose reference frame was used to 
			obtain the homography (this agent)
		j: label of the agent used to compute the homography
		E: Homography as opencv Mat
		R: vector of 1x9 with information about the possible rotation matrix the
			first decomposition
		t: vector of 1x3 with infotmation about the possible translation vector for the
			first decomposition		
*/
void Controller::IBFCE(int matches, int i, int j, Mat &E, double *R, double *t){
}

/*
	function: EBFC (Controller_type: 9)
	description: Epipole Based Formation Control
	params:
		matches: number of matches used to obtain the homography
		i: label of the agent whose reference frame was used to 
			obtain the homography (this agent)
		j: label of the agent used to compute the homography
		H: Homography as opencv Mat
		F: Fundamental matrix obtained with opencv findfundamental
*/
void Controller::EBFC(int matches,int i, int j, Mat &F){
	
	if(!matches) return;

	double p_ij[3], yaw_ij; //to store the correct result

	//--------------------------------to compute the epipole
	double Beta = K.at<double>(0,0), p[3];

	Mat W,U,Vt;
	SVD::compute(F,W,U,Vt);	
	for(int l=0;l<3;l++)
		p[l] = Vt.at<double>(2,l);

	p[2]*=Beta;
	normalize(p,3);
	
	if(!DONE[i][j]){//use the first decomposition and choice to set the filters
		Mat R_i = rotationZ(pose_i[5]);
		 Mat R_j = rotationZ(pose_j[5]);
		 Mat R_ij = R_i.t()*R_j;
				
		Vec3d p_i(pose_i[0],pose_i[1],pose_i[2]);
		 Vec3d p_j(pose_j[0],pose_j[1],pose_j[2]);
		
		Vec3f angles = rotationMatrixToEulerAngles(R_ij);
		 double yaw_ij1 = angles[2];
		
		Mat p_ij1 = R_i.t()*Mat(p_j-p_i);
		 
		p_ij[0] = p_ij1.at<double>(0,0);
		 p_ij[1] = p_ij1.at<double>(1,0);
		 p_ij[2] = p_ij1.at<double>(2,0);
		
		normalize(p_ij,3);
		
		yaw_ij = yaw_ij1;
		
		yawf[i][j] = yaw_ij1;
		
		xf[i][j] = p_ij[0];
		 yf[i][j] = p_ij[1];
		 zf[i][j] = p_ij[2];
				
		DONE[i][j] = 1;
				
	}else{//use epipoles to obtain good direction		
		filter_pose_epipoles(i,j,p,p_ij, &yaw_ij);
		
		Mat R_i = rotationZ(pose_i[5]);
		 Mat R_j = rotationZ(pose_j[5]);
		 Mat R_ij = R_i.t()*R_j;
			
		Vec3f angles = rotationMatrixToEulerAngles(R_ij);
		 yaw_ij = angles[2];
			
	}

	Vx += Kv*(p_ij[0]-x_aster[i][j]);
	Vy += Kv*(p_ij[1]-y_aster[i][j]);		
	Vz += Kv*(p_ij[2]-z_aster[i][j]);	
	Wz += Kw*(yaw_ij-yaw_aster[i][j]);	

	getError(matches,i,j,p_ij[0],p_ij[1],p_ij[2],yaw_ij);
}

/*
	function: IBFCF (controller_type: 10)
	description: Image-Based Formation Control using Features. 
		this controller is the implementation from Becerra et 
		al paper 'Vision-based Consensus' (NYP). CODE: 10
	params:
		matches: number of matches used to obtain the homography
		i: label of the agent whose reference frame was used to 
			obtain the homography (this agent)
		j: label of the agent used to compute the homography
		H: Homography as opencv Mat
*/
void Controller::IBFCF(int matches, int j,
                       vector<Point2f> &pj,
                       vector<Point2f> &pi){
	
    cout << "---------" << label << ": " << "DB0 ------------- \n" << flush;
    if(velContributions[j]==true)
        return;
    cout << "---------" << label << ": " << "DB1 ------------- \n" << flush;
    int n = pj.size();
    int m = pi.size();
    cout << "n,m = " << n << " " << m << "\n" << flush;
    
//     cout << "------------- DB1.1 ------------- \n";
//     //  varaible convertions
//     vector<Point2f> point2f_kp_j; //We define vector of point2f
//     vector<Point2f> point2f_kp_i; //We define vector of point2f
//     vector<int> mask(n,1);
//     cout << "------------- DB1.1.1 ------------- \n";
//     KeyPoint::convert(kp_j[j], point2f_kp_j, mask);
//     KeyPoint::convert(kp_i[i], point2f_kp_i, mask);
    cout << "------------- DB1.1.2 ------------- \n" << flush;
//     //Then we use this nice function from OpenCV to directly convert from KeyPoint vector to Point2f vector
    cv::Mat pointmatrix_kp_j = cv::Mat(pj).reshape(1); 
    cv::Mat pointmatrix_kp_i = cv::Mat(pi).reshape(1); 
    
    cout << "------------- DB1.2 ------------- \n" << flush;
    // Descriptor control
    double lambda = 1.0;
//     cout << pointmatrix_kp_i.dims << endl << flush;
//     cout << pointmatrix_kp_j.dims << endl << flush;
//     cout << pointmatrix_kp_i.cols << endl << flush;
//     cout << pointmatrix_kp_j.cols << endl << flush;
//     cout << pointmatrix_kp_i.rows << endl << flush;
//     cout << pointmatrix_kp_j.rows << endl << flush;
    camera_norm(pointmatrix_kp_i);
    camera_norm(pointmatrix_kp_j);
//     cout << pointmatrix_kp_j << endl << flush;
//     cout << pointmatrix_kp_i << endl << flush;
//     cout << "------------- DB1.3 ------------- \n";
//     //  Compute error for all pair kp_i kp_j
//     //  TODO : revisar que sea el eorden adecuado ij
    Mat err = pointmatrix_kp_j-pointmatrix_kp_i;
//     //  TODO : apropiate Z
//     cout << pointmatrix_kp_i << endl << flush;
    Mat L = interaction_Mat(pointmatrix_kp_i,1.0);
    double det=0.0;
    L = Moore_Penrose_PInv(L,det);
    if (det < 1e-6)
        return;
//     cout << "------------- DB1.4 ------------- \n";
    Mat U  = -1.0 * lambda * L*err.reshape(1,L.cols); 
    U = U/(float)n_neigh;
    cout << U << endl << flush;
//     cout << "------------- DB1.5 ------------- \n";
    Vx +=  U.at<float>(1,0);
	Vy +=  U.at<float>(0,0);
	Vz +=  U.at<float>(2,0);
	Wz +=  U.at<float>(5,0);
    cout << "-- Partial velocities: " << Vx << ", ";
    cout <<  Vy << ", ";
    cout <<  Vz << ", ";
    cout <<  Wz << endl << flush;
    
    cout << "------------- DB1.6 ------------- \n"<<flush;
    velContributions[j]=true;
    //  Error calculation for update
//     double p_ij [4];
//     p_ij[0] = pose_j[0] - pose_i[0];
//     p_ij[1] = pose_j[1] - pose_i[1];
//     p_ij[2] = pose_j[2] - pose_i[2];
//     p_ij[3] = pose_j[5] - pose_i[5];
//     cout << "------------- DB2 ------------- \n"<<flush;
// 	getError(matches,label,j,p_ij[0],p_ij[1],p_ij[2],p_ij[3]);
    
}

/*
	function: getError
	description: computes the feedback error and saves important data to plot.
	params:
		matches: number of matches used to obtain the homography
		i: label of the agent whose reference frame was used to 
			obtain the homography (this agent)
		j: label of the agent used to compute the homography
		hat_x: estimated x coordinate
		hat_y: estimated y coordinate
		hat_z: estimated z coordinate
		hat_yaw: estimated yaw
*/		

void Controller::getError(int matches,int i, int j, double hat_x, double hat_y, double hat_z, double hat_yaw){
	//feedback error between this two agents
	errors_t[j]=sqrt((hat_x-x_aster[i][j])*(hat_x-x_aster[i][j])+(hat_y-y_aster[i][j])*(hat_y-y_aster[i][j])+(hat_z-z_aster[i][j])*(hat_z-z_aster[i][j]));
	errors_psi[j]=abs(hat_yaw-yaw_aster[i][j]);
	
	//if we need to plot ground truth data
	string name(output_dir+"/"+to_string(j)+"/");
	
	//computing GT
	Mat R_i = rotationZ(pose_i[5]);
	Mat R_j = rotationZ(pose_j[5]);
	Mat R_ij = R_i.t()*R_j;
			
	Vec3d p_i(pose_i[0],pose_i[1],pose_i[2]);
	Vec3d p_j(pose_j[0],pose_j[1],pose_j[2]);
	
	Vec3f angles = rotationMatrixToEulerAngles(R_ij);
	double yaw_ij = angles[2];
	
	Mat p_ij = R_i.t()*Mat(p_j-p_i);
	 
	double xij = p_ij.at<double>(0,0);
	double yij = p_ij.at<double>(1,0);
	double zij = p_ij.at<double>(2,0);
	
	//estimated pose, desired pose, ground truth pose. In that order for every coordinate
	if(bearingsNeeded(controller_type)){
        double norm = sqrt(xij*xij+yij*yij+zij*zij);
        xij/=norm;
        yij/=norm;
        zij/=norm;
	}
	double data[12] =
        {hat_x,x_aster[i][j],xij,
        hat_y,y_aster[i][j],yij,
        hat_z,z_aster[i][j],zij,
        hat_yaw,yaw_aster[i][j],yaw_ij};
	
	appendToFile(name+"coordinates.txt",data,12);
	

	data[0] = (double) matches;
	appendToFile(name+"matches.txt",data,1);

	//feedback error, error against ground truth first for translation and then for rotation 
	data[0] = errors_t[j];
	data[1] = sqrt((xij-hat_x)*(xij-hat_x)+(yij-hat_y)*(yij-hat_y)+(zij-hat_z)*(zij-hat_z));	
	data[2] = errors_psi[j];
	data[3] = abs(hat_yaw-yaw_ij);
	appendToFile(name+"errors.txt",data,4);
}

/*
	function: choose_first_homography_decomposition
	description: given the information from the geometric constraint and the possible decompositions, chooses
		the one with the normal with highest Z component. This is because we suppose the agents are facing 
		the plane perpendicularly.
	params:
		matches: amount of matches used
		i: label of the agent
		j: label of the neighbor
		t: list of possible relative positions 
		r: list of possible rotation matrix
		n: list of normals
	returns: 
		1: if there is a feasible configuration
		0: there is not feasible configuration
*/
int Controller::choose_first_homography_decomposition(int matches, int i,int j,vector<Mat> &t, vector<Mat> &r, vector<Mat> &n){
	int possible = t.size(), index;
	double max = -1;

	for(int k=0;k<possible;k++){
		double value = n[k].at<double>(2,0);
		if(value > max ){
			max = value;
			index = k;
		}
	}

	xf[i][j] = t[index].at<double>(1,0); 
	yf[i][j] = t[index].at<double>(0,0);
	zf[i][j] = t[index].at<double>(2,0);
	yawf[i][j] = (double) rotationMatrixToEulerAngles(r[index])[2];

	if(yawf[i][j]!=yawf[i][j]) return 0; //its NaN
	if(matches) return 1;	//its ok
	return 0; //we have garbage in data
}

/*
	function: filter_pose
	description: Uses information from previous decompositions to choose the next solution
	params:
		i: label of the agent
		j: label of the neighbor
		r: list of possible rotation matrices
		t: list of possible relative translation vectors
		p: pointer to the array where we will save the right solution for translation
		yaw: pointer to the variable where we will save the right solution for rotation
*/
void Controller::filter_pose(int i, int j, vector<Mat> &r, vector<Mat> &t,double *p, double *yaw){
	int possible = r.size(), index;
	double min = 1000000;
	for(int k=0;k<possible;k++){
		double d[3] = {t[k].at<double>(1,0),t[k].at<double>(0,0),t[k].at<double>(2,0)};
		double x = xf[i][j], y = yf[i][j], z = zf[i][j];
		double error = sqrt((x-d[0])*(x-d[0])+(y-d[1])*(y-d[1])+(z-d[2])*(z-d[2]));
		if (error < min){
			min = error;
			index = k;
		}
	}

	p[0] = t[index].at<double>(1,0); xf[i][j] = p[0];
	p[1] = t[index].at<double>(0,0); yf[i][j] = p[1];
	p[2] = t[index].at<double>(2,0); zf[i][j] = p[2];
	*yaw = (double) rotationMatrixToEulerAngles(r[index])[2]; yawf[i][j] = *yaw;		
}

/*
	function: filter_pose_essential
	description: Uses information from previous decompositions to choose the next solution
	params:
		i: label of the agent
		j: label of the neighbor
		R1: first possible rotation
		R2: second possible rotation 
		tr: posible relative translation, another possiblity is -tr
		p: pointer to the array where we will save the right solution for translation
		yaw: pointer to the variable where we will save the right solution for rotation
*/
void Controller::filter_pose_essential(int i, int j, Mat &R1, Mat &R2, Mat &t, double *p, double *yaw){
	double d[3]; //to write it easier (xD)
	for(int k=0;k<3;k++) 
		d[k] = t.at<double>(k,0);

	p[0] = d[1]; p[1]=d[0]; p[2]= d[2];
	double error_t1 = sqrt((xf[i][j]-p[0])*(xf[i][j]-p[0])+(yf[i][j]-p[1])*(yf[i][j]-p[1])+(zf[i][j]-p[2])*(zf[i][j]-p[2]));//t
	double error_t2 = sqrt((xf[i][j]+p[0])*(xf[i][j]+p[0])+(yf[i][j]+p[1])*(yf[i][j]+p[1])+(zf[i][j]+p[2])*(zf[i][j]+p[2]));// -t

	if(error_t2<error_t1){
		p[0] = -p[0];
		p[1] = -p[1];
		p[2] = -p[2];
	}						


	double a1 = rotationMatrixToEulerAngles(R1)[2],a2 = rotationMatrixToEulerAngles(R2)[2];	
	*yaw = a1;
	if(abs(a2-yawf[i][j]) < abs(a1-yawf[i][j])) *yaw = a2;
	
	xf[i][j] = p[0];
	yf[i][j] = p[1];
	zf[i][j] = p[2];
	yawf[i][j] = *yaw;
}

/*
	function: filter_pose_epipoles
	description: Uses information from previous decompositions to choose the next solution
	params:
		i: label of the agent
		j: label of the neighbor
		e: pointer to the obtained epipole
		p: pointer to the array where we will save the right solution for translation
		yaw: pointer to the variable where we will save the right solution for rotation
*/
void Controller::filter_pose_epipoles(int i, int j, double *e, double *p, double *yaw){

	double e_i[4][3],filter[3]; filter[0] = xf[i][j];	filter[1] = yf[i][j];	filter[2] = zf[i][j];
	
	for(int l=0;l<3;l++){
		e_i[0][l] = e[l];
		e_i[1][l] = -e[l];
		e_i[2][l] = e[l];
		e_i[3][l] = -e[l];
	} 
	
	for(int l=0;l<4;l++){
		double aux = e_i[l][0];
		e_i[l][0] = e_i[l][1];
		e_i[l][1] = aux;
	}

	for(int l=0;l<2;l++){
		e_i[l][1] = -e_i[l][1];
	}
	
	double errors[4]={0,0,0,0},min=1e10;
	int min_index=0;
	
	for(int l=0;l<4;l++){
		for(int k=0;k<3;k++)
			errors[l]+=(e_i[l][k]-filter[k])*(e_i[l][k]-filter[k]);
	
		errors[l] = sqrt(errors[l]);
		if(errors[l] < min){
			min = errors[l];
			min_index = l;
		}
	}

	for(int l=0;l<3;l++)
		p[l] = e_i[min_index][l];

	*yaw = 0;
	
	xf[i][j] = p[0];
	yf[i][j] = p[1];
	zf[i][j] = p[2];
	yawf[i][j] = *yaw;
}

/*
	function: readK
	description: read the calibration matrix from the input_dir
*/
void Controller::readK(){
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
 * function:  Normalization in siut
 * description: normalices the coordinates in the matrix reference
 * params:
 *      points : Mat(n,2)
 */

void Controller::camera_norm(Mat &  points){
    
    //  Normalización in situ
    int n = points.rows;
    
    //  p1
//     cout << "-- " << label << " Norm A --\n" << flush; 
    points.col(0) = points.col(0)-K.at<double>(0,2);
    points.col(1) = points.col(1)-K.at<double>(1,2);
//     cout << "-- " << label << " Norm B --\n" << flush; 
    points.col(0) = points.col(0).mul(1.0/K.at<double>(0,0));
    points.col(1) = points.col(1).mul(1.0/K.at<double>(1,1));
//     cout << "-- " << label << " Norm C --\n" << flush; 

    return;
}

/*
 * function:  Moore Penrose Pseudoinverse matrix
 * description: computes the interaction list for a set of point descriptors
 * params:
 *      L: Input matrix
 *      det : a pointer to a double that saves the determinant calculation
 * returns 
 *      Lt : Pseudoinverse
 */

Mat  Moore_Penrose_PInv(Mat L,double & det){
    
    Mat Lt = L.t();
    Mat Ls = Lt*L;
    det = determinant(Ls);
    if (det > 1e-6){
        return Ls.inv()*Lt;
    }
        
    return Lt;
}

/*
 * function: interation matrix
 * description: computes the interaction list for a set of point descriptors
 * params:
 *      p2: OpenCV matrix with image points
 *      Z : Opencv matrix with depth aproximation
 * reutrns:
 *      L: The interaction Matrix at given state
 */
Mat interaction_Mat(Mat pointmatrix,
                double Z
                   ){
    
    int n = pointmatrix.rows;
    
    cout << "--- IM 0nm---" << n << " " << pointmatrix.cols <<  flush;

    int type_point = pointmatrix.type();
    Mat L= Mat::zeros(n,12,type_point) ;
    cout << "--- IM 1---\n" << flush;
    //  Calculos
    //   -1/Z
    L.col(0) = -Mat::ones(n,1,type_point)/Z;
//     L.col(1) =
    //  p[0,:]/Z
    L.col(2) = pointmatrix.col(0)/Z;
    //  p[0,:]*p[1,:]
    L.col(3) = pointmatrix.col(0).mul(pointmatrix.col(1));
    cout << "--- IM 2---\n" << flush;
    //  -(1+p[0,:]**2)
    L.col(4) = -1.0*(1.0+pointmatrix.col(0).mul(pointmatrix.col(0)));
    //  p[1,:]
    pointmatrix.col(1).copyTo(L.col(5));
//     L.col(6) =
    //  -1/Z
    L.col(0).copyTo(L.col(7));
    cout << "--- IM 3---\n" << flush;
    //  p[1,:]/Z
    L.col(8) = pointmatrix.col(1)/Z;
    //  1+p[1,:]**2
    L.col(9) =  1.0+pointmatrix.col(1).mul(pointmatrix.col(1));
    //  -p[0,:]*p[1,:]
    L.col(10) = -1.0*pointmatrix.col(0).mul(pointmatrix.col(1));
    //  -p[0,:]
    L.col(11) = -1.0 * pointmatrix.col(0);
    cout << "--- IM 4---\n" << flush;

    return L.reshape(1,2*n);
}
