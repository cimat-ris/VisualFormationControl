#include "image_based_formation_control/agent.h" 

using namespace fvc;



int whereis(int a , std::vector<int> v)
{
    for (int i = 0; i < v.size(); i++)
        if (a == v[i])
            return i;
    return -1;
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
    n_ArUcos_ref = nh.param(std::string("n_ArUcos_ref"),1);
    PIAG_ENABLE = nh.param(std::string("enablePIAG"),false);
    VERBOSE_ENABLE = nh.param(std::string("debug"),false);
    gamma_0 = nh.param(std::string("gamma_0"),3.);
    gamma_inf = nh.param(std::string("gamma_inf"),0.1);
    gamma_d = nh.param(std::string("gamma_d"),-5.);
    gammaIntegral_0 = nh.param(std::string("gammaIntegral_0"),0.1);
    gammaIntegral_inf = nh.param(std::string("gammaIntegral_inf"),0.01);
    gammaIntegral_d = nh.param(std::string("gammaIntegral_d"),-5.);
    
    neighbors = new int[n_agents];
    velContributions = new bool[n_agents];
    States = new vcc::state [n_agents];
    
    
    
    for (int i = 0; i < n_agents ; i++)
    {
        velContributions[i] = false;
        States[i] = tmp_State;
        // simplifiacar con (n_agents, cv::Mat())
        errors.push_back(cv::Mat());
        errors_1.push_back(cv::Mat());
        errors_2.push_back(cv::Mat());
        std::vector<std::vector<cv::Point2f>> tmp;
        complements.push_back(tmp);
        std::vector<int> tmp_id;
        complements_ids.push_back(tmp_id);
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

    //  empty data in outfiles
    std::fstream outfile(output_dir+"arUcos.dat",  std::ifstream::out | std::ifstream::trunc );
    if (!outfile.is_open() || outfile.fail())
    {
        outfile.close();
        printf("\nError : failed to erase file content !");
    }
    outfile.close();

    // std::ofstream outfile(output_dir+"error.dat", std::ifstream::out | std::ifstream::trunc );
    outfile.open(output_dir+"error.dat", std::ifstream::out | std::ifstream::trunc );
    if (!outfile.is_open() || outfile.fail())
    {
        outfile.close();
        printf("\nError : failed to erase file content !");
    }
    int row_bytes =  4*sizeof(double) +sizeof(double)+sizeof(int);
    outfile.write((char *) &row_bytes,sizeof(int));
    outfile.close();

    outfile.open(output_dir+"partial.dat", std::ifstream::out | std::ifstream::trunc );
    if (!outfile.is_open() || outfile.fail())
    {
        outfile.close();
        printf("\nError : failed to erase file content !");
    }
    row_bytes = 14*sizeof(float) +sizeof(double);
    outfile.write((char *) &row_bytes,sizeof(int));
    outfile.close();



    std::cout << "----    Data for agent " << label << "    -----"<< std::endl;
    std::cout << "Number of agents: " << n_agents << std::endl;
    std::cout << "Number of neighbors: " << n_neighbors << std::endl ;
    std::cout << "PI-AG Enable: " << PIAG_ENABLE << std::endl;
    std::cout << "Verbose Enable: " << VERBOSE_ENABLE << std::endl;
    if(PIAG_ENABLE)
    {
        std::cout << "Proportional contribution parameters: " << std::endl;
        std::cout << "Gamma_0: " << gamma_0 << std::endl;
        std::cout << "Gamma_Inf: " << gamma_inf << std::endl;
        std::cout << "Gamma_derivative: " << gamma_d << std::endl;

        std::cout << "Integral contribution parameters: " << std::endl;
        std::cout << "Gamma_0: " << gammaIntegral_0 << std::endl ;
        std::cout << "Gamma_Inf: " << gammaIntegral_inf << std::endl ;
        std::cout << "Gamma_derivative: " << gammaIntegral_d << std::endl ;
    }
    std::cout << "Output Directory: " << output_dir << std::endl ;
    std::cout << "Input Directory: " << input_dir << std::endl ;

    std::cout << "-----------------------------------------" << std::endl;

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

        cv::Ptr<cv::aruco::DetectorParameters> _parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> _dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        _parameters->adaptiveThreshWinSizeMin = 141;  // default 3
        _parameters->adaptiveThreshWinSizeMax = 251;  // default 23
        _parameters->adaptiveThreshWinSizeStep = 20;  // default 10
        _parameters->adaptiveThreshConstant = 4 ;     // default 7


        std::vector<int> _ids;
        std::vector<std::vector<cv::Point2f> > _corners;
        std::vector<std::vector<cv::Point2f> > rejected;


        cv::aruco::detectMarkers(tmp_img, _dictionary, _corners, _ids,_parameters, rejected);
        // std::cout << label << " " << _corners.size() << " ArUco search in ref\n" << std::flush;

        // cv::aruco::detectMarkers(tmp_img, _dictionary, _corners, _ids,_parameters, rejected);
        if(VERBOSE_ENABLE)
        std::cout << label << " " << _corners.size() << " ArUco search in ref\n" << std::flush;

        if(! tmp_img.empty())
        {
            if (_corners.size() == n_ArUcos_ref)
            {
                loaded_imgs++;
                aruco_refs.push_back(_corners);
                aruco_refs_ids.push_back(_ids);
            }
            else
            {
                //  Else: save empty vector
                std::vector<std::vector<cv::Point2f>> _empty;
                std::vector<int> _empty2;
                aruco_refs.push_back(_empty);
                aruco_refs_ids.push_back(_empty2);
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
    if(VERBOSE_ENABLE)
    {
        std::cout << "Init pose drone " << label << std::endl;
        std::cout << "X: " << _x << std::endl;
        std::cout << "Y: " << _y << std::endl;
        std::cout << "Z: " << _z << std::endl;
        std::cout << "Roll: " << roll << std::endl;
        std::cout << "Pitch: " << pitch << std::endl;
        std::cout << "Yaw: " << yaw << std::endl ;
        std::cout << "-------------" << std::endl;
    }
}

void fvc::agent::processImage(const sensor_msgs::Image::ConstPtr & msg)
{
    cv::Mat img;

    try{

        img=cv_bridge::toCvShare(msg,"bgr8")->image;
        std::vector<std::vector<cv::Point2f> > rejected;
        cv::aruco::detectMarkers(img, dictionary, corners, corners_ids,parameters, rejected);

    }catch (cv_bridge::Exception& e){
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }

    if (corners.size() > 0)
    {

        //  Save data:
        std::fstream outfile(output_dir+"error.dat", std::ios::app | std::ios::binary);
        for(int i = 0; i< corners_ids.size() ; i++)
        {
            outfile.write((char *) & corners_ids[i],sizeof(int));
            outfile.write((char *) & corners[i][0], 4*sizeof(cv::Point2f));
        }
        outfile.close();

        //  Draw aruco markers
        //  Draw ref
        // std::vector<std::vector<cv::Point2f>> aruco_print;
        // aruco_print.push_back(aruco_refs[label]);
        // cv::aruco::drawDetectedMarkers(img, aruco_print, cv::noArray(),cv::Scalar(0,0,255));
        cv::aruco::drawDetectedMarkers(img, aruco_refs[label], cv::noArray(),cv::Scalar(0,0,255));
        //  draw current
        cv::aruco::drawDetectedMarkers(img, corners, cv::noArray(),cv::Scalar(0,255,0));

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

        if (VERBOSE_ENABLE)
        {
            std::cout << label << " corners ids: " ;
            for (int i = 0; i < corners_ids.size(); i++)
                std::cout << corners_ids[i] << ", ";
            std::cout << std::endl << std::flush;
        }

    }else{
        std::cout << label << " -- " << " ArUco(s) lost\n" << std::flush;
//             ARUCO_COMPUTED = false;

    }
}

void fvc::agent::getImageDescription(const image_based_formation_control::corners::ConstPtr& msg)
{
    int j = msg->j;
    int size = msg->size;
    //  clear complements
    complements[j].clear();
    std::vector<std::vector<cv::Point2f>> _complements(size);
    std::vector<int> ids(size);

    for (int i = 0; i<size; i++)
    {

        std::vector<cv::Point2f> complement(4);
        int ref_id = msg->ArUcos[i].id;

        //  consenso

        complement[0].x = msg->ArUcos[i].points[0].x;
        complement[0].y = msg->ArUcos[i].points[0].y;
        complement[1].x = msg->ArUcos[i].points[1].x;
        complement[1].y = msg->ArUcos[i].points[1].y;
        complement[2].x = msg->ArUcos[i].points[2].x;
        complement[2].y = msg->ArUcos[i].points[2].y;
        complement[3].x = msg->ArUcos[i].points[3].x;
        complement[3].y = msg->ArUcos[i].points[3].y;


        //      then add point and partial error (-p*_i -(p_j - p*_j))
        int idx = whereis(ref_id,aruco_refs_ids[j] );
        int idx_label = whereis(ref_id,aruco_refs_ids[label] );
        if (idx <0 || idx_label <0)
            continue;

        for (int k = 0; k< 4; k++)
        {
            complement[k] = complement[k] + aruco_refs[label][idx_label][k];
            complement[k] = complement[k] - aruco_refs[j][idx][k];
        }
        _complements[i] = complement;
        ids[i] = ref_id;
    }

    complements[j] = _complements;
    complements_ids[j] = ids;
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

    //Save consensus error
    std::string name = output_dir+"partial.dat";
    States[label].save_data(time, name);
    name = output_dir+"error.dat";

    std::ofstream outfile(name, std::ios::app | std::ios::binary);
    for (int i = 0; i < ArUcos_ovelap.size(); i++)
    {
        cv::Mat tmp;
        errors[label](cv::Rect(0,i*4,2,4 )).copyTo(tmp);
        outfile.write((char *) &time,sizeof(double));
        outfile.write((char *) &ArUcos_ovelap[i],sizeof(int));
        outfile.write((char *) tmp.data,tmp.elemSize() * tmp.total());

    }
    outfile.close();
}

//  allows to start the simulation when the pose is updated
bool fvc::agent::isUpdated()
{
    return POSITION_UPDATED;
}

image_based_formation_control::corners fvc::agent::getArUco( ){
    image_based_formation_control::corners corners_msg;
    corners_msg.j = label;
    corners_msg.size = corners_ids.size();

    if(ARUCO_COMPUTED)
    {
        for (int i = 0; i < corners_ids.size(); i++)
        {
            image_based_formation_control::ArUco ArUco_msg;
            ArUco_msg.id = corners_ids[i];
            ArUco_msg.points[0].x = corners[i][0].x;
            ArUco_msg.points[0].y = corners[i][0].y;
            ArUco_msg.points[1].x = corners[i][1].x;
            ArUco_msg.points[1].y = corners[i][1].y;
            ArUco_msg.points[2].x = corners[i][2].x;
            ArUco_msg.points[2].y = corners[i][2].y;
            ArUco_msg.points[3].x = corners[i][3].x;
            ArUco_msg.points[3].y = corners[i][3].y;

            corners_msg.ArUcos.push_back(ArUco_msg);
        }
    }else{
        std::cout << "ERR: Null ArUco was send" << std::endl << std::flush;

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


std::vector<int> fvc::agent::getIdxArUco(int a, std::vector<std::vector<int>> list)
{
    std::vector<int> selector;
    for (int i = 0; i< n_agents; i++)
    {
        if (isNeighbor(i))
        {
            int tmp = whereis(a, list[i]);
            if (tmp > -1)
                selector.push_back(tmp);
            else
            {
                std::vector<int> None;
                return None;
            }
        }else{
            selector.push_back(-1);
        }
    }
    return selector;
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
    
    std::vector<cv::Point2f> _corners;
    std::vector<std::vector<cv::Point2f>> _complements;
    for (int i = 0; i< n_agents; i++)
    {
        std::vector<cv::Point2f> tmp;
        _complements.push_back( tmp);
    }

    //  check correspondence
    ArUcos_ovelap.clear();


    for (int j = 0;j < corners_ids.size(); j++)
    {
        //  se crea la lista de índices que tienen el aruco corners_ids[j]
        std::vector<int> selector = getIdxArUco(corners_ids[j],complements_ids);

        if (selector.size() > 0)
        {
            // _corners.push_back(corners[j]);
            std::vector<cv::Point2f>::iterator begin = corners[j].begin();
            std::vector<cv::Point2f>::iterator end = corners[j].end();
            std::vector<cv::Point2f>::iterator self_end = _corners.end();
            _corners.insert (self_end,begin,end);
            ArUcos_ovelap.push_back(corners_ids[j]);
            for (int i = 0; i< n_agents; i++)
            {
                if (isNeighbor(i))
                {
                    // lista de índices para el mismo aruco por cada agente
                    std::vector<cv::Point2f>  tmp = complements[i][selector[i]];

                    // _complements[i].push_back(tmp);

                    std::vector<cv::Point2f>::iterator tbegin = tmp.begin();
                    std::vector<cv::Point2f>::iterator tend = tmp.end();
                    std::vector<cv::Point2f>::iterator tself_end = _complements[i].end();
                    _complements[i].insert(tself_end, tbegin, tend);
                }
            }
        }
    }

    if (ArUcos_ovelap.size() ==0)
    {
        return;
    }

    for (int i = 0; i< n_agents; i++)
    {
        if (isNeighbor(i))
        {
            
            //  real  control
            //  Add velocity contribution
            cv::Mat tmp1 = cv::Mat(_corners).reshape(1);
            cv::Mat tmp2 = cv::Mat(_complements[i]).reshape(1);
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
    cv::Mat p_current_int = cv::Mat(_corners).reshape(1);

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
    {
        for (int i = 0; i < corners_ids.size(); i++)
        errors_integral.push_back(cv::Mat::zeros(4,2,result.p1.type()  ));
        // errors_integral = cv::Mat::zeros(result.p1.size(),result.p1.type()  );

        INTEGRAL_INIT = true;
    }

    //  estimación de Z
    int n = result.p2.rows;
    int type = result.p2.type();
    cv::Mat Z = cv::Mat::ones(n,1,type);
    Z = States[label].Z * Z;
    
    //  Interaction matrix calculation 
    // cv::Mat p_current_int = cv::Mat(corners).reshape(1);
    cv::Mat p_current_n;
    p_current_int.copyTo(p_current_n);
    vcc::camera_norm(States[label].params, p_current_n);
    cv::Mat L = vcc::interaction_Mat(p_current_n,Z);

    double det=0.0;
    L = vcc::Moore_Penrose_PInv(L,det);
    if (det < 1e-14)
    {
        std::cout << label << " --- Controller error" << std::endl;
        // return;

    }
    if(VERBOSE_ENABLE)
    {
        std::cout << label << " -- det = " << det << std::endl << std::flush;
        std::cout << "Error:\n" << errors[label] << std::endl << std::flush;
    }

    cv::Mat U;

    if(PIAG_ENABLE)
    {
        cv::Mat int_tmp;

        //  TODO: join int_tmp
        for (int i = 0; i < ArUcos_ovelap.size(); i++)
        {
            int j = whereis(ArUcos_ovelap[i], aruco_refs_ids[label]);
            int_tmp.push_back(errors_integral[j]);
        }

        double gamma =  adaptGamma(gamma_0, gamma_inf, gamma_d, errors[label]);
        double gammaIntegral =  adaptGamma(gammaIntegral_0, gammaIntegral_inf, gammaIntegral_d, int_tmp);
        U = L*(gamma * errors[label].reshape(1,L.cols) + gammaIntegral * int_tmp.reshape(1,L.cols) );
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

//  TODO: generalize for N-Arucos
void fvc::agent::integrateError(double dt)
{
         // errors_integral *= 0.95;
         // errors_integral += dt * errors[label];

    for (int i = 0; i< ArUcos_ovelap.size();i++)
    {
        int j = whereis(ArUcos_ovelap[i],aruco_refs_ids[label]);
        errors_integral[j] *= 0.95;
        errors_integral[j] += dt * errors[label](cv::Rect(0,i*4,2,4 ));
    }
}


