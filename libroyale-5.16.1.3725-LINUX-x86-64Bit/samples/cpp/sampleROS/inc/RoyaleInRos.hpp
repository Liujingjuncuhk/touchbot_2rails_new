/****************************************************************************\
 * Copyright (C) 2023 pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#include <royale.hpp>

#include <thread>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_msgs/UInt16.h>
#include <std_msgs/UInt32.h>

#define ROYALE_ROS_MAX_STREAMS 2u

class RoyaleInRos : public nodelet::Nodelet, public royale::IDepthDataListener, public royale::IExposureListener {
  public:
    RoyaleInRos();
    ~RoyaleInRos();

    // Overridden from Nodelet
    void onInit() override;

    // Starting and stopping the camera
    void start();
    void stop();

  private:
    // Called by Royale for every new frame
    void onNewData(const royale::DepthData *data) override;

    // Will be called when the newly calculated exposure time deviates from currently set exposure time of the current UseCase
    void onNewExposure(const uint32_t exposureTime, const royale::StreamId streamId) override;

    // Create cameraInfo, return true if the setting is successful, otherwise false
    bool setCameraInfo();

    // Calculate the FPS
    void fpsUpdate();

    void initUseCase();

    // The required parameters for UI's initialization
    void initMsgUpdate();

    // Callback a bool value to determine whether the UI is initialized
    void callbackIsInit(const std_msgs::Bool::ConstPtr &msg);

    // Callback the changed settings from UI and set the changed settings
    void callbackUseCase(const std_msgs::String::ConstPtr &msg);
    void callbackExpoTime(const std_msgs::UInt32::ConstPtr &msg, uint32_t streamIdx);
    void callbackMaxExpoTime(const std_msgs::UInt32::ConstPtr &msg);
    void callbackExpoMode(const std_msgs::Bool::ConstPtr &msg, uint32_t streamIdx);
    void callbackMinFiler(const std_msgs::Float32::ConstPtr &msg);
    void callbackMaxFiler(const std_msgs::Float32::ConstPtr &msg);
    void callbackDivisor(const std_msgs::UInt16::ConstPtr &msg);
    void callbackParams(const std_msgs::String::ConstPtr &msg, uint32_t streamIdx);

    sensor_msgs::CameraInfo m_cameraInfo;
    std_msgs::String m_msgInitPanel;
    std_msgs::String m_msgExpoTimeParam[ROYALE_ROS_MAX_STREAMS];
    std_msgs::UInt32 m_msgExpoTimeValue[ROYALE_ROS_MAX_STREAMS];

    ros::Publisher m_pubCameraInfo;
    ros::Publisher m_pubCloud[ROYALE_ROS_MAX_STREAMS];
    ros::Publisher m_pubDepth[ROYALE_ROS_MAX_STREAMS];
    ros::Publisher m_pubGray[ROYALE_ROS_MAX_STREAMS];

    ros::Publisher m_pubInit;
    ros::Publisher m_pubExpoTimeParam[ROYALE_ROS_MAX_STREAMS];
    ros::Publisher m_pubExpoTimeValue[ROYALE_ROS_MAX_STREAMS];
    ros::Publisher m_pubFps[ROYALE_ROS_MAX_STREAMS];

    ros::Subscriber m_subIsInit;
    ros::Subscriber m_subUseCase;
    ros::Subscriber m_subExpoTime[ROYALE_ROS_MAX_STREAMS];
    ros::Subscriber m_subMaxExpoTime;
    ros::Subscriber m_subExpoMode[ROYALE_ROS_MAX_STREAMS];
    ros::Subscriber m_subMinFilter;
    ros::Subscriber m_subMaxFilter;
    ros::Subscriber m_subDivisor;
    ros::Subscriber m_subParams[ROYALE_ROS_MAX_STREAMS];

    std::unique_ptr<royale::ICameraDevice> m_cameraDevice;
    royale::Pair<uint32_t, uint32_t> m_limits[ROYALE_ROS_MAX_STREAMS];

    std::string m_useCases;
    std::thread m_fpsProcess;
    uint64_t m_frames[ROYALE_ROS_MAX_STREAMS];
    std::string m_cam_expected_id;
    std::string m_cam_name;
    std::string m_node_name;
    std::string m_cam_access_code;
    std::string m_startUseCase;
    uint32_t m_exposureTime[ROYALE_ROS_MAX_STREAMS];
    uint16_t m_grayDivisor;
    float m_minFilter;
    float m_maxFilter;
    bool m_initPanel;
    bool m_autoExposure[ROYALE_ROS_MAX_STREAMS];
    bool m_inv_z;
    std::string m_recording_file;
    royale::String m_currentUseCase;
    royale::Vector<royale::StreamId> m_streamIds;
};
