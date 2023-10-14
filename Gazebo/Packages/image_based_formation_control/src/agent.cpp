#include "image_based_formation_control/agent.h" 

using namespace fvc;

static void save_mat(double time, std::string directory, cv::Mat & _mat)
{
    std::ofstream outfile;
    outfile.open(directory, std::ios_base::app);	
    outfile << time << " " ;
    
    float * mat_ptr = _mat.ptr<float>(0,0);
    int n = _mat.cols * _mat.rows;
    for(int i=0;i<n;i++)
		outfile << mat_ptr[i] << " " ;
	outfile << std::endl;
    
    outfile.close();
}

fvc::agent::agent(std::string name)
{
    label = atoi(name.c_str());
}

fvc::agent::~agent()
{
    if (neighbors != nullptr)
        delete neighbors;
    
    if (velContributions != nullptr)
        delete velContributions;
    
    if (States != nullptr)
        delete States;
    
}

void fvc::agent::load(const ros::NodeHandle &nh)
{
//     State.load(nh);
    vcc::state tmp_State;
    tmp_State.load(nh);
    // Load Laplacian rows
    n_agents = nh.param(std::string("n_agents"),0);
    PIAG_ENABLE = nh.param(std::string("enablePIAG"),false);
    VERBOSE_ENABLE = nh.param(std::string("debug"),false);
    gamma_0 = nh.param(std::string("gamma_0"),false);
    gamma_inf = nh.param(std::string("gamma_inf"),false);
    gamma_d = nh.param(std::string("gamma_d"),false);
    gammaIntegral_0 = nh.param(std::string("gammaIntegral_0"),false);
    gammaIntegral_inf = nh.param(std::string("gammaIntegral_inf"),false);
    gammaIntegral_d = nh.param(std::string("gammaIntegral_d"),false);
    
    neighbors = new int[n_agents];
    velContributions = new bool[n_agents];
    States = new vcc::state [n_agents];
    
    
    
    for (int i = 0; i < n_agents ; i++)
    {
        velContributions[i] = false;
        States[i] = tmp_State;
        errors.push_back(cv::Mat());
        errors_1.push_back(cv::Mat());
        errors_2.push_back(cv::Mat());
        std::vector<cv::Point2f> tmp;
        complements.push_back(tmp);
    }
    
    int i = label*n_agents;
    int end = i + n_agents;
    
    XmlRpc::XmlRpcValue kConfig;
    if (nh.hasParam("Laplacian")) {
        nh.getParam("Laplacian", kConfig);
        if (kConfig.getType() == XmlRpc::XmlRpcValue::TypeArray)
        {
            //  TODO: simplify
            int idx = 0;
            for (;i<end;i++) {
                std::ostringstream ostr;
                ostr << kConfig[i];
                std::istringstream istr(ostr.str());
                istr >>  neighbors[idx] ;
                idx++;
            }
        }
    }
    n_neighbors = -1*neighbors[label];
//     nh.getParam("input_dir",input_dir);
    nh.param<std::string>("input_dir",input_dir,"");
    if(input_dir.empty())
    {
        std::cout << "Empty string in input directory \n";
        exit(EXIT_FAILURE);
    }
//     nh.getParam("output_dir",output_dir);
    nh.param<std::string>("output_dir",output_dir,"");
    if(output_dir.empty())
    {
        std::cout << "Empty string in output directory \n";
        exit(EXIT_FAILURE);
    }
    output_dir = output_dir + std::to_string(label);
    output_dir = output_dir + "/";
    std::cout << input_dir << std::endl;
    std::cout << output_dir << std::endl;

    //  TODO enable PIAG
    //  TODO enable verbose
}

//  reads the reference images
//      returns true if the number of read images are the
//      same as the number of agents
bool fvc::agent::imageRead()
{
    int loaded_imgs = 0;
    
    //  load a referenc for each agent
    //      sef inluded
    for (int i = 0; i < n_agents; i++)
    {
        std::string name = input_dir+"reference"+std::to_string(i)+".png";
        cv::Mat tmp_img = cv::imread(name, cv::IMREAD_COLOR);
        
        //  ArUco detection in reference
        //  TODO: Optimize
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > _corners;
        std::vector<std::vector<cv::Point2f> > rejected;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    //     aruco::detectMarkers(img, dictionary, _corners, ids);
        cv::aruco::detectMarkers(tmp_img, dictionary, _corners, ids,parameters, rejected);
        std::cout << _corners.size() << " ArUco search in ref\n" << std::flush;

        
        if(! tmp_img.empty())
        {
            
            if (_corners.size() > 0) 
            {
                loaded_imgs++;
                
                //  add aruco refs interface
                
                //  If corners got found, save them
                aruco_refs.push_back(_corners[0]);
            }
            else
            {
                //  Else: save empty vector
                std::vector<cv::Point2f> _empty;
                aruco_refs.push_back(_empty);
            }
        
            
            //  Append
            desired_img.push_back(tmp_img);
        }
        
    }
    
    
    return (loaded_imgs == n_agents);
}

void fvc::agent::setPose(const geometry_msgs::Pose::ConstPtr& msg){	
	if (POSITION_UPDATED) return;

    //creating quaternion
    tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
    //creatring rotation matrix ffrom quaternion
    tf::Matrix3x3 rot_mat(q);
    //obtaining euler angles
    double _x, _y, _z;
    double roll, pitch, yaw;
    rot_mat.getEulerYPR(yaw, pitch, roll);
    //saving the data obtained
    _x = msg->position.x;
    _y = msg->position.y;
    _z = msg->position.z;


//         State.initialize(_x,_y,_z,yaw);
    for (int i = 0; i < n_agents; i ++)
        States[i].initialize(_x,_y,_z,yaw);

    POSITION_UPDATED = true;

    std::cout << "Init pose drone " << label << std::endl;
    std::cout << "X: " << _x << std::endl;
    std::cout << "Y: " << _y << std::endl;
    std::cout << "Z: " << _z << std::endl;
    std::cout << "Roll: " << roll << std::endl;
    std::cout << "Pitch: " << pitch << std::endl;
    std::cout << "Yaw: " << yaw << std::endl ;
    std::cout << "-------------" << std::endl;

}

void fvc::agent::processImage(const sensor_msgs::Image::ConstPtr & msg)
{
    try{		
//         processor.takePicture(cv_bridge::toCvShare(msg,"bgr8")->image);
//         processor.detectAndCompute(pose);		
        cv::Mat img=cv_bridge::toCvShare(msg,"bgr8")->image;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > _corners;
        std::vector<std::vector<cv::Point2f> > rejected;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    //     aruco::detectMarkers(img, dictionary, _corners, ids);
        cv::aruco::detectMarkers(img, dictionary, _corners, ids,parameters, rejected);
//         std::cout << label << " -- " << _corners.size() << " ArUco search\n" << std::flush;
        
        
        if (_corners.size() > 0)
        {
        
            //  Save message
//             std::cout << label << " -- " << _corners.size() << " ArUco(s) detected\n" << std::flush;
            corners = _corners[0];
            
//             if(ARUCO_COMPUTED)
//                 corners.clear();
//             
//             for (int i = 0; i < 4 ; i++)
//                 corners.push_back(_corners[0][i]);
            //  Draw aruco markers
            //  Draw ref
            std::vector<std::vector<cv::Point2f>> aruco_print;
            aruco_print.push_back(aruco_refs[label]);
            cv::aruco::drawDetectedMarkers(img, aruco_print, cv::noArray(),cv::Scalar(0,0,255));
            //  draw current
            cv::aruco::drawDetectedMarkers(img, _corners, cv::noArray(),cv::Scalar(0,255,0));
            
            //  image mesgae contructor
            image_msg = cv_bridge::CvImage(
                std_msgs::Header(),
                sensor_msgs::image_encodings::BGR8,img).toImageMsg();
            image_msg->header.frame_id = "matching_image";
            image_msg->width = img.cols;
            image_msg->height = img.rows;
            image_msg->is_bigendian = false;
            image_msg->step = sizeof(unsigned char) * img.cols*3;
            image_msg->header.stamp = ros::Time::now();
            
            
            ARUCO_COMPUTED = true;
        }else{
            std::cout << label << " -- " << " ArUco(s) lost\n" << std::flush;
//             ARUCO_COMPUTED = false;
            
        }

    }catch (cv_bridge::Exception& e){
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void fvc::agent::getImageDescription(const image_based_formation_control::corners::ConstPtr& msg)
{
    int j = msg->j;
    
    std::vector<cv::Point2f> complement(4);
    
    //  consenso
    
    complement[0].x = msg->data[0].x;
    complement[0].y = msg->data[0].y;
    complement[1].x = msg->data[1].x;
    complement[1].y = msg->data[1].y;
    complement[2].x = msg->data[2].x;
    complement[2].y = msg->data[2].y;
    complement[3].x = msg->data[3].x;
    complement[3].y = msg->data[3].y;
    
    //      then add point and partial error (-p*_i -(p_j - p*_j))
    for (int i = 0; i< 4; i++)
    {
        complement[i] = complement[i] + aruco_refs[label][i];
        complement[i] = complement[i] - aruco_refs[j][i];
    }
        
    complements[j] = complement;
    
//     //  Case for reference only
//     complement = aruco_refs[label];

        velContributions[j] = true;
}

bool fvc::agent::isNeighbor(int j)
{
    if (j == label) return false;
    if (neighbors[j] == 0 ) return false;
    return true;
}

void fvc::agent::save_state(double time)
{
//     std::cout << output_dir << std::endl;
//     State.save_data(time,output_dir+"state.txt");
    for(int i = 0; i< n_agents; i++)
    {
        std::string name = output_dir+"partial_"+std::to_string(i)+".txt";
        States[i].save_data(time, name);
        name = output_dir+"error_"+std::to_string(i)+".txt";
        save_mat(time,name,errors[i]);
    }
    //Save consensus error
}

//  allows to start the simulation when the pose is updated
bool fvc::agent::isUpdated()
{
    return POSITION_UPDATED;
}

image_based_formation_control::corners fvc::agent::getArUco(){
    image_based_formation_control::corners corners_msg;
	corners_msg.j = label;
    if(ARUCO_COMPUTED)
    {
        corners_msg.data[0].x = corners[0].x;
        corners_msg.data[0].y = corners[0].y;
        corners_msg.data[1].x = corners[1].x;
        corners_msg.data[1].y = corners[1].y;
        corners_msg.data[2].x = corners[2].x;
        corners_msg.data[2].y = corners[2].y;
        corners_msg.data[3].x = corners[3].x;
        corners_msg.data[3].y = corners[3].y;
    }else{
        std::cout << "ERR: Null ArUco was send" << std::endl << std::flush;
        corners_msg.data[0].x = 0.;
        corners_msg.data[0].y = 0.;
        corners_msg.data[1].x = 0.;
        corners_msg.data[1].y = 0.;
        corners_msg.data[2].x = 0.;
        corners_msg.data[2].y = 0.;
        corners_msg.data[3].x = 0.;
        corners_msg.data[3].y = 0.;
    }
	return corners_msg;
}

bool fvc::agent::incompleteComputedVelocities(){
    int count = 0;
    for (int i = 0 ; i < n_agents; i++)
    {
        if(isNeighbor(i))
        if(velContributions[i] == true)
            count++;
    }
    
    if (count == n_neighbors)
        return false;
    
    return true;
    
}

void fvc::agent::execControl(double dt)
{
    if(!ARUCO_COMPUTED) return;

    States[label].Vx = 0.;
    States[label].Vy = 0.;
    States[label].Vz = 0.;
    States[label].Vroll = 0.;
    States[label].Vpitch = 0.;
    States[label].Vyaw = 0.;
    
    for (int i = 0; i < n_agents ; i++)
    {
        errors[i] = cv::Mat();
        errors_1[i] = cv::Mat();
        errors_2[i] = cv::Mat();
    }
    
    for (int i = 0; i< n_agents; i++)
    {
        if (isNeighbor(i))
        {
            
            //  real  control
            //  Add velocity contribution
            cv::Mat tmp1 = cv::Mat(corners).reshape(1);
            cv::Mat tmp2 = cv::Mat(complements[i]).reshape(1);
//             std::cout << label << tmp1 << tmp2 << std::endl << std::flush;
            //  save error 
            tmp2.copyTo(errors_2[i]);
            tmp1.copyTo(errors_1[i]);
            
            if (errors_2[label].empty())
                errors_2[i].copyTo(errors_2[label]);
            else
                errors_2[label] = errors_2[label] + errors_2[i];
            
            if (errors_1[label].empty())
                errors_1[i].copyTo(errors_1[label]);
            else
                errors_1[label] = errors_1[label] + errors_1[i];
            
            
            
        }
    }
    



    //  Set up result
    vcc::matching_result result;
    errors_1[label].copyTo(result.p2);
    errors_2[label].copyTo(result.p1);
    
    //  Normalization
    vcc::camera_norm(States[label].params, result);
    
    //  Err = p2 - p1
    errors[label] = result.p1 - result.p2;
    
    //  INIT integral r c t
    if (!INTEGRAL_INIT)
    errors_integral = cv::Mat::zeros(result.p1.size(),result.p1.type()  );
    
    //  estimación de Z
    int n = result.p2.rows;
    int type = result.p2.type();
    cv::Mat Z = cv::Mat::ones(n,1,type);
    Z = States[label].Z * Z;
    
    //  Interaction matrix calculation 
    cv::Mat p_current_int = cv::Mat(corners).reshape(1);
    cv::Mat p_current;
    p_current_int.copyTo(p_current);
    vcc::camera_norm(States[label].params, p_current);
    cv::Mat L = vcc::interaction_Mat(p_current,Z);

    double det=0.0;
    L = vcc::Moore_Penrose_PInv(L,det);
    if (det < 1e-14)
    {
        std::cout << label << " --- Controller error" << std::endl;
        // return;

    }
    
    std::cout << label << " -- det = " << det << std::endl << std::flush;

    cv::Mat U;

    if(PIAG_ENABLE)
    {
        double gamma =  adaptGamma(gamma_0, gamma_inf, gamma_d, errors[label]);
        double gammaIntegral =  adaptGamma(gammaIntegral_0, gammaIntegral_inf, gammaIntegral_d, errors_integral);
        U = L*(gamma * errors[label].reshape(1,L.cols) + gammaIntegral * errors_integral.reshape(1,L.cols) );
        U /= (float) n_neighbors ;
        U *= -1.;

        integrateError( dt);
    }else{
        U = - 1.*  L*errors[label].reshape(1,L.cols) / (float) n_neighbors ;
    }
    
    /**********Updating velocities in the axis*/
    //velocities from homography decomposition
//      U = cv::Mat(U,CV_32F);
    States[label].Vx += (float) U.at<float>(1,0);
    States[label].Vy += (float) U.at<float>(0,0);
    States[label].Vz += (float) U.at<float>(2,0);
    States[label].Vyaw += (float) U.at<float>(3,0);
    // States[label].Vyaw += (float) U.at<float>(5,0); // 6DOF
    

    

    //  TODO: Simpla state update V2
    States[label].update();


}

trajectory_msgs::MultiDOFJointTrajectory fvc::agent::getPose()
{
    //create message for the pose
    Eigen::VectorXd position; 
    position.resize(3); 
    position(0) = States[label].X; 
    position(1) = States[label].Y; 
    position(2) = States[label].Z;
//     position(0) = State.X; 
//     position(1) = State.Y; 
//     position(2) = State.Z;
    
    // prepare msg
    trajectory_msgs::MultiDOFJointTrajectory position_msg;
    position_msg.header.stamp=ros::Time::now();
    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(
        position, States[label].Yaw, &position_msg);
//         position, State.Yaw, &position_msg);
    
    return position_msg;
	
}

void fvc::agent::reset(char SELECT)
{
    if (SELECT & CONTRIBUTIONS)
    {
        for (int i = 0; i < n_agents ; i++)
        {
            States[i].Vx = 0.;
            States[i].Vy = 0.;
            States[i].Vz = 0.;
            States[i].Vyaw = 0.;
            States[i].Vpitch = 0.;
            States[i].Vroll = 0.;
            
            errors[i] = cv::Mat();
            velContributions[i] = false;
        }

    }
    
    
    if (SELECT & CORNERS)
    {
        ARUCO_COMPUTED = false;
    }
}



double fvc::agent::adaptGamma(double _gamma_0, double _gamma_inf, double _gamma_d,
                          cv::Mat _error)
{
    //
    double gamma =  cv::norm(_error );
    gamma *=  _gamma_d ;
    gamma /=  _gamma_0 - _gamma_inf;
    gamma =  exp(gamma);
    gamma *=   _gamma_0 - _gamma_inf;
    gamma += _gamma_inf;
    return gamma;
}

void fvc::agent::integrateError(double dt)
{
         errors_integral += dt * errors[label];
}


