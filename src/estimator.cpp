#include "estimator.h"

#include "backend/vertex_inverse_depth.h"
#include "backend/vertex_pose.h"
#include "backend/vertex_speedbias.h"
#include "backend/edge_reprojection.h"
#include "backend/edge_imu.h"

#include <ostream>
#include <fstream>

using namespace myslam;

Estimator::Estimator() : f_manager{Rs}
{
    // ROS_INFO("init begins");

    for (size_t i = 0; i < WINDOW_SIZE + 1; i++)
    {
        pre_integrations[i] = nullptr;
    }
    for(auto &it: all_image_frame)
    {
        it.second.pre_integration = nullptr;
    }
    tmp_pre_integration = nullptr;
    
    clearState();
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        // cout << "1 Estimator::setParameter tic: " << tic[i].transpose()
        //     << " ric: " << ric[i] << endl;
    }
    cout << "1 Estimator::setParameter FOCAL_LENGTH: " << FOCAL_LENGTH << endl;
    f_manager.setRic(ric);
    project_sqrt_info_ = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    
    tmp_pre_integration = nullptr;
    
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

//采用的是中值法计算IMU预积分
void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;//作为上一时刻的线加速度和角速度
        gyr_0 = angular_velocity;
    }
    
    // IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];    
    // 是一个数组，里面存放着(WINDOW_SIZE + 1)个指针，指针指向的类型是IntegrationBase    
    // 如果是新的一帧, 则新建一个预积分项目, WINDOW_SIZE原项目里是10
    // 该指针元数是否为空，WINDOW_SIZE=10
    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // frame_count是窗内图像帧的计数    
    // 一个窗内有多个相机帧，每个相机帧之间又有多个IMU数据 
    if (frame_count != 0)
    {
        //JST: 这里有名堂！push_back()这个成员函数最后调用了另一个成员函数propagate()，且原封不动地把自己的三个输入参数都传递了过去
        //JST:  integration_base.h内midPointIntegration函数，考虑了误差传递
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);//IntegrationBase类型, 输入到图像中的预积分值??

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);
        //使用中值法求解当前时刻PVQ对应公式，用线加速度和角速度对位置、旋转、速度积分
        //JST:此处积分没有误差传递，但是之前的pre_integrations里面是考虑的，这里的Rs,Ps,Vs给谁用啊？
        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double header)
{
    //ROS_DEBUG("new image coming ------------------------------------------");
    // cout << "Adding feature points: " << image.size()<<endl;
    // 判断次新帧是否是关键帧   
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD; // 是关键帧，marg掉老帧 
    else
        marginalization_flag = MARGIN_SECOND_NEW; // 不是关键帧，marg掉前一帧

    //ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    //ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    //ROS_DEBUG("Solving %d", frame_count);
    // cout << "number of feature: " << f_manager.getFeatureCount()<<endl;
    Headers[frame_count] = header;

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(header, imageframe));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if (ESTIMATE_EXTRINSIC == 2)
    {
        cout << "calibrating extrinsic param, rotation movement is needed" << endl;
        if (frame_count != 0)
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                // ROS_WARN("initial extrinsic rotation calib success");
                // ROS_WARN_STREAM("initial extrinsic rotation: " << endl
                                                            //    << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL)
    {
        if (frame_count == WINDOW_SIZE)
        {
            bool result = false;
            if (ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
            {
                // cout << "1 initialStructure" << endl;
                result = initialStructure();
                initial_timestamp = header;
            }
            if (result)
            {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                f_manager.removeFailures();
                cout << "Initialization finish!" << endl;
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];
            }
            else
                slideWindow();
        }
        else
            frame_count++;
    }
    else
    {
        TicToc t_solve;
        solveOdometry();
        //ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            // ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            // ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        f_manager.removeFailures();
        //ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if (var < 0.25)
        {
            // ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    //判断当前的图像帧是否产生了足够了平移，能够计算相对位姿态
    if (!relativePose(relative_R, relative_T, l))
    {
        cout << "Not enough features or parallax; Move device around" << endl;
        return false;
    }
    GlobalSFM sfm;
    if (!sfm.construct(frame_count + 1, Q, T, l,
                       relative_R, relative_T,
                       sfm_f, sfm_tracked_points))
    {
        cout << "global SFM failed!" << endl;
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin();
    for (int i = 0; frame_it != all_image_frame.end(); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame_it->first) == Headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if ((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = -R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6)
        {
            cout << "Not enough points for solve pnp pts_3_vector size " << pts_3_vector.size() << endl;
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            cout << " solve pnp fail!" << endl;
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp, tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    //初始窗口中的图像帧完成SFM三维重建之后，即各图像帧在参考坐标系下的初始位姿都已经计算完成之后，执行的
    if (visualInitialAlign())
        return true;
    else
    {
        cout << "misalign visual structure with IMU" << endl;
        return false;
    }
}


bool Estimator::visualInitialAlign()
{
    // 1. 计算陀螺仪偏置，尺度，重力加速度和速度
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result)
    {
        //ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    // 2. 传递所有图像帧的位姿Ps、Rs，并将其置为关键帧 
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R;
        Vector3d Pi = all_image_frame[Headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    // 3. 重新计算所有f_manager的特征点深度  
    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;//将所有特征点的深度置为-1
    f_manager.clearDepth(dep);

    //重新计算特征点的深度  
    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero();
    //RIC中存放的是相机到IMU的旋转，在相机-IMU外参标定部分求得
    ric[0] = RIC[0];
    f_manager.setRic(ric);
    // 三角化计算地图点的深度
    // Ps中存放的是各个帧相对于参考帧之间的平移，RIC[0]为相机-IMU之间的旋转 
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));


    // 5. 将Ps、Vs、depth尺度s缩放后从l帧转变为相对于c0帧图像坐标系下
    /**           
    * 前面初始化中，计算出来的是相对滑动窗口中第l帧的位姿，在这里转换到IMU bo坐标系下           
    * s*p_bk^​b0​​=s*p_bk^​cl​​−s*p_b0^​cl​​=(s*p_ck^​cl​​−R_bk​^cl​​*p_c^b​)−(s*p_c0^​cl​​−R_b0​^cl​​*p_c^b​)           
    * TIC[0]是相机到IMU的平移量           
    * Rs是IMU第k帧到滑动窗口中图像第l帧的旋转           
    * Ps是滑动窗口中第k帧到第l帧的平移量           
    * 注意：如果运行的脚本是配置文件中无外参的脚本，那么这里的TIC都是0          
    */    
    // (1) 位移Ps    
    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    // Ps转变为第i帧imu坐标系到第0帧imu坐标系的变换
    // 之前相机第l帧为参考系，转换到IMU bo为基准坐标系
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    
    // （2）速度Vs
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if (frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }
    // (3) 深度
    // 更新每个地图点被观测到的帧数(used_num)和预测的深度(estimated_depth)
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }
    /**
     * refine之后就获得了C_0坐标系下的重力g^{c_0}，此时通过将g^{c_0}旋转至z轴方向，
     * 这样就可以计算相机系到世界坐标系的旋转矩阵q_{c_0}^w，这里求得的是rot_diff,这样就可以将所有变量调整至世界系中。
    */    
    // Rs是IMU第k帧到滑动窗口中图像第l帧的旋转
    // R0将参考坐标系旋转到z轴垂直向上 
    // R2ypr：旋转矩阵或四元数 到 欧拉角
    // ypr2R：欧拉角 到 旋转矩阵或四元数
    Matrix3d R0 = Utility::g2R(g);
    // R0将参考系的y轴旋转到第0帧的IMU正前方，这个时候x轴也确定了向右
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    //  相机系到世界坐标系的旋转矩阵 R0
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 只考虑偏航角的影响   
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();

    //7.所有变量从参考坐标系c_l旋转到世界坐标系w 
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    //ROS_DEBUG_STREAM("g0     " << g.transpose());
    //ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, WINDOW_SIZE);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;
            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                //ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR)
    {
        TicToc t_tri;
        f_manager.triangulate(Ps, tic, ric);
        //cout << "triangulation costs : " << t_tri.toc() << endl;        
        backendOptimization();
    }
}
/*
此函数将向量转变成数组指针的形式，生成的优化变量包括：
para_Pose（6维，相机位姿）、
para_SpeedBias（9维，相机速度、加速度偏置、角速度偏置）、
para_Ex_Pose（6维、相机IMU外参）、
para_Feature（1维，特征点深度）、
para_Td（1维，标定同步时间）
在后面进行边缘化操作时这些优化变量都是当做整体看待。
*/
void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5])
                                             .toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        //ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5])
                               .toRotationMatrix()
                               .transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) +
                origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5])
                     .toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info)
    {
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) +
                 origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;
    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2)
    {
        //ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        //ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        //ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        //ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        //ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::MargOldFrame()
{
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    // 外参也是待估计变量？？
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        //R,t
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    //WINDOW_SIZE=10
    //设置待估计变量
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        // 滑动窗口内每一帧的位姿，PQ 
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        // 滑窗内每一帧的IMU 的速度v 加速度ba、角速度的偏置 bg 
        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();

        //逆深度见后
    }

    // IMU
    {
        if (pre_integrations[1]->sum_dt < 10.0)
        {   
            //设置IMU预积分的残差
            std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[1]));
            std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
            edge_vertex.push_back(vertexCams_vec[0]);
            edge_vertex.push_back(vertexVB_vec[0]);
            edge_vertex.push_back(vertexCams_vec[1]);
            edge_vertex.push_back(vertexVB_vec[1]);
            imuEdge->SetVertex(edge_vertex);
            problem.AddEdge(imuEdge);
        }
    }

    // Visual Factor
    // 包括获取每个特征的逆深度以及对于与同一个路标点共视的所有帧，计算第一帧(最老帧)与各帧之间的重投影误差、信息矩阵…
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            //待估计逆深度
            shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX inv_d(1);
            inv_d << para_Feature[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);

            // 遍历所有的观测
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Vector3d pts_j = it_per_frame.point;
                
                // 对于与同一个路标点共视的所有帧，计算第一帧(最老帧)与各帧之间的重投影误差  
                std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);// 逆深度
                edge_vertex.push_back(vertexCams_vec[imu_i]);// 两帧的PQ   
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                edge_vertex.push_back(vertexExt); // 外参

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem，固定先验信息矩阵lambda_p?
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
        }
    }

    //传入要边缘化掉的最老帧信息,进行边缘化
    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    marg_vertex.push_back(vertexCams_vec[0]); //这里的0即为最老帧对应的索引 // PQ 
    marg_vertex.push_back(vertexVB_vec[0]);   // V,bias 
    problem.Marginalize(marg_vertex, pose_dim); // 边缘化处理,传入要边缘化的顶点和所有状态向量的维度和。
    Hprior_ = problem.GetHessianPrior();
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}
void Estimator::MargNewFrame()
{

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    //    vector<backend::Point3d> points;
    int pose_dim = 0;

    //以下内容，同MargOldFrame
    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);

            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
        else
        {
            Hprior_ = MatXX(pose_dim, pose_dim);
            Hprior_.setZero();
            bprior_ = VecX(pose_dim);
            bprior_.setZero();
        }
    }

    std::vector<std::shared_ptr<backend::Vertex>> marg_vertex;
    // 把窗口倒数第二个帧 marg 掉
    marg_vertex.push_back(vertexCams_vec[WINDOW_SIZE - 1]);//注意这里的索引，次新帧
    marg_vertex.push_back(vertexVB_vec[WINDOW_SIZE - 1]);
    problem.Marginalize(marg_vertex, pose_dim);
    Hprior_ = problem.GetHessianPrior();
    bprior_ = problem.GetbPrior();
    errprior_ = problem.GetErrPrior();
    Jprior_inv_ = problem.GetJtPrior();
}
void Estimator::problemSolve()
{
    backend::LossFunction *lossfunction;
    lossfunction = new backend::CauchyLoss(1.0);
    //    lossfunction = new backend::TukeyLoss(1.0);

    // step1. 构建 problem
    backend::Problem problem(backend::Problem::ProblemType::SLAM_PROBLEM);
    vector<shared_ptr<backend::VertexPose>> vertexCams_vec;
    vector<shared_ptr<backend::VertexSpeedBias>> vertexVB_vec;
    int pose_dim = 0;

    // 先把 外参数 节点加入图优化，这个节点在以后一直会被用到，所以我们把他放在第一个
    shared_ptr<backend::VertexPose> vertexExt(new backend::VertexPose());
    {
        Eigen::VectorXd pose(7);
        pose << para_Ex_Pose[0][0], para_Ex_Pose[0][1], para_Ex_Pose[0][2], para_Ex_Pose[0][3], para_Ex_Pose[0][4], para_Ex_Pose[0][5], para_Ex_Pose[0][6];
        vertexExt->SetParameters(pose);

        if (!ESTIMATE_EXTRINSIC)
        {
            //ROS_DEBUG("fix extinsic param");
            // TODO:: set Hessian prior to zero
            vertexExt->SetFixed();
        }
        else{
            //ROS_DEBUG("estimate extinsic param");
        }
        problem.AddVertex(vertexExt);
        pose_dim += vertexExt->LocalDimension();
    }

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        shared_ptr<backend::VertexPose> vertexCam(new backend::VertexPose());
        Eigen::VectorXd pose(7);
        pose << para_Pose[i][0], para_Pose[i][1], para_Pose[i][2], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5], para_Pose[i][6];
        vertexCam->SetParameters(pose);
        vertexCams_vec.push_back(vertexCam);
        problem.AddVertex(vertexCam);
        pose_dim += vertexCam->LocalDimension();

        shared_ptr<backend::VertexSpeedBias> vertexVB(new backend::VertexSpeedBias());
        Eigen::VectorXd vb(9);
        vb << para_SpeedBias[i][0], para_SpeedBias[i][1], para_SpeedBias[i][2],
            para_SpeedBias[i][3], para_SpeedBias[i][4], para_SpeedBias[i][5],
            para_SpeedBias[i][6], para_SpeedBias[i][7], para_SpeedBias[i][8];
        vertexVB->SetParameters(vb);
        vertexVB_vec.push_back(vertexVB);
        problem.AddVertex(vertexVB);
        pose_dim += vertexVB->LocalDimension();
    }

    // IMU
    for (int i = 0; i < WINDOW_SIZE; i++)
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;

        std::shared_ptr<backend::EdgeImu> imuEdge(new backend::EdgeImu(pre_integrations[j]));
        std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
        edge_vertex.push_back(vertexCams_vec[i]);
        edge_vertex.push_back(vertexVB_vec[i]);
        edge_vertex.push_back(vertexCams_vec[j]);
        edge_vertex.push_back(vertexVB_vec[j]);
        imuEdge->SetVertex(edge_vertex);
        problem.AddEdge(imuEdge);
    }

    // Visual Factor
    vector<shared_ptr<backend::VertexInverseDepth>> vertexPt_vec;
    {
        int feature_index = -1;
        // 遍历每一个特征
        for (auto &it_per_id : f_manager.feature)
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size();
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            shared_ptr<backend::VertexInverseDepth> verterxPoint(new backend::VertexInverseDepth());
            VecX inv_d(1);
            inv_d << para_Feature[feature_index][0];
            verterxPoint->SetParameters(inv_d);
            problem.AddVertex(verterxPoint);
            vertexPt_vec.push_back(verterxPoint);

            // 遍历所有的观测
            for (auto &it_per_frame : it_per_id.feature_per_frame)
            {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                Vector3d pts_j = it_per_frame.point;

                std::shared_ptr<backend::EdgeReprojection> edge(new backend::EdgeReprojection(pts_i, pts_j));
                std::vector<std::shared_ptr<backend::Vertex>> edge_vertex;
                edge_vertex.push_back(verterxPoint);
                edge_vertex.push_back(vertexCams_vec[imu_i]);
                edge_vertex.push_back(vertexCams_vec[imu_j]);
                edge_vertex.push_back(vertexExt);

                edge->SetVertex(edge_vertex);
                edge->SetInformation(project_sqrt_info_.transpose() * project_sqrt_info_);

                edge->SetLossFunction(lossfunction);
                problem.AddEdge(edge);
            }
        }
    }

    // 先验
    {
        // 已经有 Prior 了
        if (Hprior_.rows() > 0)
        {
            // 外参数先验设置为 0. TODO:: 这个应该放到 solver 里去弄
            //            Hprior_.block(0,0,6,Hprior_.cols()).setZero();
            //            Hprior_.block(0,0,Hprior_.rows(),6).setZero();

            problem.SetHessianPrior(Hprior_); // 告诉这个 problem
            problem.SetbPrior(bprior_);
            problem.SetErrPrior(errprior_);
            problem.SetJtPrior(Jprior_inv_);
            problem.ExtendHessiansPriorSize(15); // 但是这个 prior 还是之前的维度，需要扩展下装新的pose
        }
    }

    problem.Solve(10);

    // update bprior_,  Hprior_ do not need update
    if (Hprior_.rows() > 0)
    {
        std::cout << "----------- update bprior -------------\n";
        std::cout << "             before: " << bprior_.norm() << std::endl;
        std::cout << "                     " << errprior_.norm() << std::endl;
        bprior_ = problem.GetbPrior();
        errprior_ = problem.GetErrPrior();
        std::cout << "             after: " << bprior_.norm() << std::endl;
        std::cout << "                    " << errprior_.norm() << std::endl;
    }

    // update parameter
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        VecX p = vertexCams_vec[i]->Parameters();
        for (int j = 0; j < 7; ++j)
        {
            para_Pose[i][j] = p[j];
        }

        VecX vb = vertexVB_vec[i]->Parameters();
        for (int j = 0; j < 9; ++j)
        {
            para_SpeedBias[i][j] = vb[j];
        }
    }

    // 遍历每一个特征
    for (int i = 0; i < vertexPt_vec.size(); ++i)
    {
        VecX f = vertexPt_vec[i]->Parameters();
        para_Feature[i][0] = f[0];
    }
}

void Estimator::backendOptimization()
{
    TicToc t_solver;
    // 借助 vins 框架，维护变量
    vector2double();
    // 构建求解器
    problemSolve();
    // 优化后的变量处理下自由度
    double2vector();
    //ROS_INFO("whole time for solver: %f", t_solver.toc());

    // 维护 marg
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        vector2double();

        MargOldFrame();

        std::unordered_map<long, double *> addr_shift; // prior 中对应的保留下来的参数地址
        // 从1开始，因为第一帧的状态要剔除
        for (int i = 1; i <= WINDOW_SIZE; i++)// 整体前移一格，去掉最老帧
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];//  外参保持不变
        if (ESTIMATE_TD)
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
        }
    }
    else
    {
        if (Hprior_.rows() > 0)
        {

            vector2double();

            MargNewFrame();

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];
            }
        }
    }
    
}


void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                Headers[i] = Headers[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                it_0->second.pre_integration = nullptr;

                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it)
                {
                    if (it->second.pre_integration)
                        delete it->second.pre_integration;
                    it->second.pre_integration = NULL;
                }

                all_image_frame.erase(all_image_frame.begin(), it_0);
                all_image_frame.erase(t_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i];
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew();
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}
