/****************************************************************************\
 * Copyright (C) 2023 pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#include <RoyaleInRos.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <sstream>
#include <stdlib.h>

using namespace std;
using namespace royale;

RoyaleInRos::RoyaleInRos() : IDepthDataListener(),
                             IExposureListener(),
                             m_cameraDevice(nullptr),
                             m_cam_expected_id(""),
                             m_cam_name(""),
                             m_node_name(""),
                             m_cam_access_code(""),
                             m_startUseCase(""),
                             m_grayDivisor(100),
                             m_minFilter(0.0f),
                             m_maxFilter(7.5f),
                             m_initPanel(false),
                             m_inv_z(false),
                             m_recording_file(""),
                             m_currentUseCase("") {
    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        m_frames[i] = 0u;
        m_autoExposure[i] = true;
        m_exposureTime[i] = 0;
    }
}

RoyaleInRos::~RoyaleInRos() {
    stop();
}

void RoyaleInRos::onInit() {
    // Retrieve the handle of the current node
    auto nodeHandle = getMTPrivateNodeHandle();

    // Check if we want to open a specific camera
    if (nodeHandle.hasParam("royale_cam_id")) {
        nodeHandle.getParam("royale_cam_id", m_cam_expected_id);
        if (!(m_cam_expected_id.empty()))
            NODELET_INFO_STREAM("Specific camera requested: <" << m_cam_expected_id << ">");
    }

    // Check if we provided a Royale access code
    if (nodeHandle.hasParam("royale_access_code")) {
        nodeHandle.getParam("royale_access_code", m_cam_access_code);
        if (!(m_cam_access_code.empty()))
            NODELET_INFO_STREAM("Access code provided: <" << m_cam_access_code << ">");
    }

    if (nodeHandle.hasParam("node_name")) {
        nodeHandle.getParam("node_name", m_node_name);
        if (!(m_node_name.empty()))
            NODELET_INFO_STREAM("Node name: <" << m_node_name << ">");
    }

    if (nodeHandle.hasParam("royale_use_case")) {
        nodeHandle.getParam("royale_use_case", m_startUseCase);
        if (!(m_startUseCase.empty()))
            NODELET_INFO_STREAM("Start use case: <" << m_startUseCase << ">");
    }

    m_cam_name = m_node_name;
    std::replace(m_cam_name.begin(), m_cam_name.end(), '-', '_');

    std::stringstream royale_cam_node_str_stream;
    royale_cam_node_str_stream << "/royale_cam_" << m_cam_name;
    std::string royale_cam_node_str = royale_cam_node_str_stream.str();

    // Check if a filename for recording was specified
    if (nodeHandle.hasParam("royale_cam_record")) {
        nodeHandle.getParam("royale_cam_record", m_recording_file);
        NODELET_INFO_STREAM("Recording file : <" << m_recording_file << ">");
    } else {
        m_recording_file = "";
    }

    // Check if we should invert the z axis
    if (nodeHandle.hasParam("rc_inv_z")) {
        int rc_inv_z = 0;
        nodeHandle.getParam("rc_inv_z", rc_inv_z);
        m_inv_z = (1 == rc_inv_z);
    }
    if (m_inv_z)
        NODELET_INFO_STREAM("Attention: inverting Z axis in the point cloud.");
    else
        NODELET_DEBUG_STREAM("Not inverting Z axis in the point cloud.");

    // Advertise the camera info
    m_pubCameraInfo = nodeHandle.advertise<sensor_msgs::CameraInfo>(royale_cam_node_str + "/camera_info", 1);

    // Advertise our point cloud topic and image topics for every stream
    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        m_pubCloud[i] = nodeHandle.advertise<sensor_msgs::PointCloud2>(royale_cam_node_str + "/point_cloud_" + std::to_string(i), 1);
        m_pubDepth[i] = nodeHandle.advertise<sensor_msgs::Image>(royale_cam_node_str + "/depth_image_" + std::to_string(i), 1);
        m_pubGray[i] = nodeHandle.advertise<sensor_msgs::Image>(royale_cam_node_str + "/gray_image_" + std::to_string(i), 1);
    }

    // Advertise the topics to initialize and update the settings of camera in the UI
    m_pubInit = nodeHandle.advertise<std_msgs::String>(royale_cam_node_str + "/init_panel", 1);

    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        m_pubExpoTimeParam[i] = nodeHandle.advertise<std_msgs::String>(royale_cam_node_str + "/expo_time_param_" + std::to_string(i), 1);
        m_pubExpoTimeValue[i] = nodeHandle.advertise<std_msgs::UInt32>(royale_cam_node_str + "/expo_time_value_" + std::to_string(i), 1);
        m_pubFps[i] = nodeHandle.advertise<std_msgs::String>(royale_cam_node_str + "/update_fps_" + std::to_string(i), 1);

        // Subscribe to the exposure time and exposure mode messages to receive updates from the GUI
        m_subExpoTime[i] = nodeHandle.subscribe<std_msgs::UInt32::ConstPtr, const std_msgs::UInt32::ConstPtr &>(royale_cam_node_str + "/expo_time_" + std::to_string(i), 1, [this, i](const auto &msg) { callbackExpoTime(msg, i); });
        m_subExpoMode[i] = nodeHandle.subscribe<std_msgs::Bool::ConstPtr, const std_msgs::Bool::ConstPtr &>(royale_cam_node_str + "/expo_mode_" + std::to_string(i), 1, [this, i](const auto &msg) { callbackExpoMode(msg, i); });

        m_subParams[i] = nodeHandle.subscribe<std_msgs::String::ConstPtr, const std_msgs::String::ConstPtr &>(royale_cam_node_str + "/params_" + std::to_string(i), 1, [this, i](const auto &msg) { callbackParams(msg, i); });
    }

    // Subscribe the messages of UI to change the state of the camera
    m_subIsInit = nodeHandle.subscribe(royale_cam_node_str + "/is_init", 1, &RoyaleInRos::callbackIsInit, this);
    m_subUseCase = nodeHandle.subscribe(royale_cam_node_str + "/use_case", 1, &RoyaleInRos::callbackUseCase, this);
    m_subMaxExpoTime = nodeHandle.subscribe(royale_cam_node_str + "/expo_max_time", 1, &RoyaleInRos::callbackMaxExpoTime, this);
    m_subMinFilter = nodeHandle.subscribe(royale_cam_node_str + "/min_filter", 1, &RoyaleInRos::callbackMinFiler, this);
    m_subMaxFilter = nodeHandle.subscribe(royale_cam_node_str + "/max_filter", 1, &RoyaleInRos::callbackMaxFiler, this);
    m_subDivisor = nodeHandle.subscribe(royale_cam_node_str + "/divisor", 1, &RoyaleInRos::callbackDivisor, this);

    start();
}

void RoyaleInRos::start() {
    m_fpsProcess = thread(&RoyaleInRos::fpsUpdate, this);

    // Create a camera manager and query available cameras
    CameraManager manager(m_cam_access_code);
    Vector<String> cameraList(manager.getConnectedCameraList());

    if (cameraList.empty()) {
        NODELET_ERROR_STREAM("No suitable cameras found!");
        return;
    }

    for (const auto &x : cameraList)
        NODELET_INFO_STREAM("Found camera : " << x);

    // Create the first camera that was found, register a data listener
    // and start the capturing
    if (m_cam_expected_id.empty())
        m_cam_expected_id = cameraList[0].toStdString();
    NODELET_DEBUG_STREAM("Creating camera : " << m_cam_expected_id.c_str());
    m_cameraDevice = manager.createCamera(m_cam_expected_id); // (cameraList[0]);
    if (nullptr == m_cameraDevice) {
        NODELET_ERROR_STREAM("No such camera found!");
        return;
    }
    NODELET_DEBUG_STREAM("Created camera : " << m_cam_expected_id.c_str());

    String cameraName;
    m_cameraDevice->getCameraName(cameraName);
    String camId;
    m_cameraDevice->getId(camId);
    NODELET_INFO_STREAM("Opened camera : " << cameraName.c_str() << ", ID: " << camId);

    if (m_cameraDevice->initialize(m_startUseCase) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Error initializing the camera!");
        return;
    }

    if (m_cameraDevice->getStreams(m_streamIds) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't retrieve streams!");
        return;
    }

    if (!setCameraInfo()) {
        NODELET_ERROR_STREAM("Couldn't create camera info!");
        return;
    }

    if (m_cameraDevice->registerExposureListener(this) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't register exposure listener!");
        return;
    }

    if (m_cameraDevice->registerDataListener(this) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't register data listener!");
        return;
    }

    if (m_cameraDevice->startCapture() != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Error starting camera capture!");
        return;
    }

    for (auto i = 0u; i < m_streamIds.size(); ++i) {
        if (m_cameraDevice->setExposureMode(ExposureMode::AUTOMATIC, m_streamIds[i]) != CameraStatus::SUCCESS) {
            NODELET_ERROR_STREAM("Couldn't set automatic exposure!");
            return;
        }
    }

    initUseCase();

    // Check which use cases of this camera are supported
    std::stringstream ss;
    Vector<String> useCases;
    m_cameraDevice->getUseCases(useCases);
    for (size_t i = 0; i < useCases.size(); ++i) {
        uint32_t streamCount;
        if (m_cameraDevice->getNumberOfStreams(useCases[i], streamCount) != CameraStatus::SUCCESS) {
            return;
        } else {
            if (streamCount <= ROYALE_ROS_MAX_STREAMS) {
                ss << "/" << useCases[i];
            }
        }
    }
    m_useCases = ss.str();
    NODELET_INFO_STREAM("Supported use cases: " << m_useCases);

    // If we specified a file start the recording
    if (!(m_recording_file.empty())) {
        char resolvedPath[PATH_MAX];
        realpath(m_recording_file.c_str(), resolvedPath);
        NODELET_INFO_STREAM("Activating recording to <" << resolvedPath << ">");
        m_cameraDevice->startRecording(m_recording_file);
    } else
        NODELET_INFO_STREAM("Recording not requested.");

    NODELET_INFO_STREAM("Royale ROS node initialized!");
    initMsgUpdate();
}

void RoyaleInRos::stop() {
    // Close the camera
    if (m_cameraDevice &&
        m_cameraDevice->stopCapture() != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Error stopping camera capture!");
        return;
    }

    m_fpsProcess.join();
}

void RoyaleInRos::onNewData(const DepthData *data) {
    // Check the idx of this stream
    auto curIdx = 0u;
    for (auto i = 0u; i < m_streamIds.size(); ++i) {
        if (data->streamId == m_streamIds[i]) {
            curIdx = i;
            break;
        }
    }

    // Make sure the UI is reinitialized each time when rviz is started
    if (m_pubFps[0].getNumSubscribers() == 0) {
        m_initPanel = false;
    }

    // Initialize the UI
    if (!m_initPanel) {
        m_pubInit.publish(m_msgInitPanel);

        for (auto i = 0u; i < m_streamIds.size(); ++i) {
            m_pubExpoTimeParam[i].publish(m_msgExpoTimeParam[i]);
            m_pubExpoTimeValue[i].publish(m_msgExpoTimeValue[i]);
        }
    }

    m_frames[curIdx]++;

    auto numPoints = data->getNumPoints();

    // Create a standard header
    std_msgs::Header header;
    std::stringstream royale_cam_node_str_stream;
    royale_cam_node_str_stream << "royale_camera_" << m_node_name << "_optical_frame";
    header.frame_id = royale_cam_node_str_stream.str();
    header.seq = 0;
    header.stamp.fromNSec(chrono::duration_cast<chrono::nanoseconds>(data->timeStamp).count());

    // Create camera info message
    sensor_msgs::CameraInfoPtr msgCameraInfo = sensor_msgs::CameraInfoPtr(new sensor_msgs::CameraInfo);

    *msgCameraInfo = m_cameraInfo;
    msgCameraInfo->header = header;
    msgCameraInfo->height = data->height;
    msgCameraInfo->width = data->width;

    // Create point cloud message ...
    sensor_msgs::PointCloud2Ptr msgPointCloud = sensor_msgs::PointCloud2Ptr(new sensor_msgs::PointCloud2);

    // ... where we want to save x,y,z
    msgPointCloud->fields.resize(3);

    msgPointCloud->fields[0].name = "x";
    msgPointCloud->fields[0].offset = 0;
    msgPointCloud->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
    msgPointCloud->fields[0].count = 1;

    msgPointCloud->fields[1].name = "y";
    msgPointCloud->fields[1].offset = static_cast<uint32_t>(sizeof(float));
    msgPointCloud->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
    msgPointCloud->fields[1].count = 1;

    msgPointCloud->fields[2].name = "z";
    msgPointCloud->fields[2].offset = 2u * static_cast<uint32_t>(sizeof(float));
    msgPointCloud->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
    msgPointCloud->fields[2].count = 1;

    msgPointCloud->header = header;
    msgPointCloud->width = data->width;
    msgPointCloud->height = data->height;
    msgPointCloud->is_bigendian = false;
    msgPointCloud->is_dense = false;
    msgPointCloud->point_step = static_cast<uint32_t>(3 * sizeof(float));
    msgPointCloud->row_step = static_cast<uint32_t>(3 * sizeof(float) * data->width);

    // Reserve space for the actual data
    msgPointCloud->data.resize(3 * sizeof(float) * numPoints);

    // Create a point cloud modifier
    sensor_msgs::PointCloud2Modifier modifier(*msgPointCloud);
    modifier.setPointCloud2FieldsByString(1, "xyz");

    // Create iterators for the three fields in our point cloud
    sensor_msgs::PointCloud2Iterator<float> iterX(*msgPointCloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iterY(*msgPointCloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iterZ(*msgPointCloud, "z");

    // Create Depth Image message
    sensor_msgs::ImagePtr msgDepthImage = sensor_msgs::ImagePtr(new sensor_msgs::Image);

    msgDepthImage->header = header;
    msgDepthImage->width = data->width;
    msgDepthImage->height = data->height;
    msgDepthImage->is_bigendian = false;
    msgDepthImage->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    msgDepthImage->step = static_cast<uint32_t>(sizeof(float) * data->width);
    msgDepthImage->data.resize(sizeof(float) * numPoints);

    float *iterDepth = (float *)&msgDepthImage->data[0];

    // Create Gray Image message
    sensor_msgs::ImagePtr msgGrayImage = sensor_msgs::ImagePtr(new sensor_msgs::Image);

    msgGrayImage->header = header;
    msgGrayImage->width = data->width;
    msgGrayImage->height = data->height;
    msgGrayImage->is_bigendian = false;
    msgGrayImage->encoding = sensor_msgs::image_encodings::MONO16;
    msgGrayImage->step = static_cast<uint32_t>(sizeof(uint16_t) * data->width);
    msgGrayImage->data.resize(sizeof(uint16_t) * numPoints);

    uint16_t *iterGray = (uint16_t *)&msgGrayImage->data[0];

    // Iterate over all the points we received in the callback
    for (auto i = 0u; i < numPoints; ++i) {
        if (data->hasDepth && data->getDepthConfidence(i) > 0.0f // Check confidence
            && data->getZ(i) >= m_minFilter && data->getZ(i) <= m_maxFilter) {
            *iterX = data->getX(i);
            *iterY = data->getY(i);
            *iterZ = m_inv_z ? -data->getZ(i) : data->getZ(i);
            *iterDepth = data->getZ(i);
        } else {
            // If the confidence is 0 set this point to NaN
            // (according to http://www.ros.org/reps/rep-0117.html)
            *iterX = *iterY = *iterZ = numeric_limits<float>::quiet_NaN();
            *iterDepth = 0.0f;
        }

        // Set divisor of gray image to adjust the brightness
        uint16_t clampedVal = std::min(m_grayDivisor, data->getGrayValue(i));
        int newGrayValue = std::min(254, static_cast<int>(254 * 1.f * (float)clampedVal / (float)m_grayDivisor));

        if (newGrayValue < 0) {
            newGrayValue = 0;
        }

        *iterGray = static_cast<uint16_t>(newGrayValue);

        ++iterX;
        ++iterY;
        ++iterZ;
        ++iterDepth;
        ++iterGray;
    }

    // Publish the messages
    m_pubCameraInfo.publish(msgCameraInfo);
    m_pubCloud[curIdx].publish(msgPointCloud);
    m_pubDepth[curIdx].publish(msgDepthImage);
    m_pubGray[curIdx].publish(msgGrayImage);
}

void RoyaleInRos::onNewExposure(const uint32_t exposureTime, const royale::StreamId streamId) {
    // Check the idx of this stream
    auto curIdx = 0u;
    for (auto i = 0u; i < m_streamIds.size(); ++i) {
        if (streamId == m_streamIds[i]) {
            curIdx = i;
            break;
        }
    }

    if (m_exposureTime[curIdx] == exposureTime) {
        return;
    }
    m_exposureTime[curIdx] = exposureTime;

    // Publish the new exposure time to the UI
    m_msgExpoTimeValue[curIdx].data = exposureTime;

    if (m_autoExposure[curIdx]) {
        m_pubExpoTimeValue[curIdx].publish(m_msgExpoTimeValue[curIdx]);
    }
}

bool RoyaleInRos::setCameraInfo() {
    LensParameters lensParams;
    if ((m_cameraDevice->getLensParameters(lensParams) == CameraStatus::SUCCESS)) {
        if (lensParams.distortionRadial.size() != 3) {
            NODELET_ERROR_STREAM("Unknown distortion model!");
            return false;
        } else {
            m_cameraInfo.distortion_model = "plumb_bob";
            m_cameraInfo.D.resize(5);
            m_cameraInfo.D[0] = lensParams.distortionRadial[0];
            m_cameraInfo.D[1] = lensParams.distortionRadial[1];
            m_cameraInfo.D[2] = lensParams.distortionTangential.first;
            m_cameraInfo.D[3] = lensParams.distortionTangential.second;
            m_cameraInfo.D[4] = lensParams.distortionRadial[2];
        }

        m_cameraInfo.K[0] = lensParams.focalLength.first;
        m_cameraInfo.K[1] = 0;
        m_cameraInfo.K[2] = lensParams.principalPoint.first;
        m_cameraInfo.K[3] = 0;
        m_cameraInfo.K[4] = lensParams.focalLength.second;
        m_cameraInfo.K[5] = lensParams.principalPoint.second;
        m_cameraInfo.K[6] = 0;
        m_cameraInfo.K[7] = 0;
        m_cameraInfo.K[8] = 1;

        m_cameraInfo.R[0] = 1;
        m_cameraInfo.R[1] = 0;
        m_cameraInfo.R[2] = 0;
        m_cameraInfo.R[3] = 0;
        m_cameraInfo.R[4] = 1;
        m_cameraInfo.R[5] = 0;
        m_cameraInfo.R[6] = 0;
        m_cameraInfo.R[7] = 0;
        m_cameraInfo.R[8] = 1;

        m_cameraInfo.P[0] = lensParams.focalLength.first;
        m_cameraInfo.P[1] = 0;
        m_cameraInfo.P[2] = lensParams.principalPoint.first;
        m_cameraInfo.P[3] = 0;
        m_cameraInfo.P[4] = 0;
        m_cameraInfo.P[5] = lensParams.focalLength.second;
        m_cameraInfo.P[6] = lensParams.principalPoint.second;
        m_cameraInfo.P[7] = 0;
        m_cameraInfo.P[8] = 0;
        m_cameraInfo.P[9] = 0;
        m_cameraInfo.P[10] = 1;
        m_cameraInfo.P[11] = 0;

        return true;
    } else {
        NODELET_ERROR_STREAM("Couldn't get lens parameters!");
        return false;
    }
}

void RoyaleInRos::fpsUpdate() {
    while (ros::ok()) {
        this_thread::sleep_for(chrono::seconds(1));

        for (auto i = 0u; i < m_streamIds.size(); ++i) {
            // Publish the fps to the UI
            std_msgs::String msg;
            msg.data = to_string(m_frames[i]);
            m_pubFps[i].publish(msg);

            m_frames[i] = 0;
        }
    }
}

void RoyaleInRos::initMsgUpdate() {
    std::stringstream ss;
    ss << m_currentUseCase << "/";
    ss << to_string(m_minFilter) << "/";
    ss << to_string(m_maxFilter) << "/";
    ss << to_string(m_grayDivisor);
    ss << m_useCases;

    m_msgInitPanel.data = ss.str();
}

void RoyaleInRos::callbackIsInit(const std_msgs::Bool::ConstPtr &msg) {
    m_initPanel = msg->data;
}

void RoyaleInRos::initUseCase() {
    std::stringstream ss;

    if (m_cameraDevice->getCurrentUseCase(m_currentUseCase) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't retrieve current use case!");
        return;
    }

    if (m_cameraDevice->getStreams(m_streamIds) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't retrieve streams!");
        return;
    }

    for (auto i = 0u; i < m_streamIds.size(); ++i) {
        // Whenever use case is switched, get the current range and value of exposure time
        // and publish the message to the UI
        m_cameraDevice->getExposureLimits(m_limits[i], m_streamIds[i]);
        std::stringstream ss;
        ss << to_string(m_limits[i].first) << "/" << to_string(m_limits[i].second);

        ExposureMode expoMode;
        m_cameraDevice->getExposureMode(expoMode, m_streamIds[i]);
        if (expoMode == ExposureMode::AUTOMATIC) {
            m_autoExposure[i] = true;
        } else {
            m_autoExposure[i] = false;
        }
        ss << "/" << to_string(m_autoExposure[i]);
        m_msgExpoTimeParam[i].data = ss.str();
        ss.str("");

        m_pubExpoTimeParam[i].publish(m_msgExpoTimeParam[i]);
        m_pubExpoTimeValue[i].publish(m_msgExpoTimeValue[i]);
    }
}

void RoyaleInRos::callbackUseCase(const std_msgs::String::ConstPtr &msg) {
    // The use case was switched in the GUI
    String currentMode = msg->data.c_str();
    CameraStatus ret = m_cameraDevice->setUseCase(currentMode);
    if (ret != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't set use case - requested was: (" << currentMode << "), return code was: (" << ret << ")!");
        return;
    } else {
        NODELET_INFO_STREAM("Successfully activated use case: " << currentMode);

        initUseCase();
        initMsgUpdate();
    }
}

void RoyaleInRos::callbackMaxExpoTime(const std_msgs::UInt32::ConstPtr &msg) {
    // Sets the maximum exposure time for the automatic exposure algorithm
    NODELET_DEBUG_STREAM("Trying to set max expo time: " << msg->data);

    const CameraStatus ret = m_cameraDevice->setProcessingParameters({{"exposureLimitUpper", royale::Variant(static_cast<int>(msg->data))}}, m_streamIds[0]);
    if (ret != CameraStatus::SUCCESS) {
        NODELET_WARN_STREAM("Couldn't set max. exposure time to: " << msg->data << ", return code " << ret);
        return;
    }
    NODELET_INFO_STREAM("Did it! I set the max expo time to: " << msg->data);
}

void RoyaleInRos::callbackExpoTime(const std_msgs::UInt32::ConstPtr &msg, uint32_t streamIdx) {
    m_exposureTime[streamIdx] = msg->data;

    // If the device is busy we try to set the exposure "tries" times
    int tries = 5;
    CameraStatus ret;
    do {
        ret = m_cameraDevice->setExposureTime(m_exposureTime[streamIdx], m_streamIds[streamIdx]);
        if (ret == CameraStatus::DEVICE_IS_BUSY) {
            this_thread::sleep_for(chrono::milliseconds(200));
            tries--;
        }
    } while (tries > 0 && ret == CameraStatus::DEVICE_IS_BUSY);

    if (ret != CameraStatus::SUCCESS) {
        NODELET_WARN_STREAM("Couldn't set exposure time!");
        return;
    }
}

void RoyaleInRos::callbackExpoMode(const std_msgs::Bool::ConstPtr &msg, uint32_t streamIdx) {
    m_autoExposure[streamIdx] = msg->data;
    ExposureMode newMode;
    if (m_autoExposure[streamIdx]) {
        newMode = ExposureMode::AUTOMATIC;
        NODELET_DEBUG_STREAM("Trying to activate auto exposure!");
    } else {
        newMode = ExposureMode::MANUAL;
        NODELET_DEBUG_STREAM("Trying to deactivate auto exposure!");
    }
    if (m_cameraDevice->setExposureMode(newMode, m_streamIds[streamIdx]) != CameraStatus::SUCCESS) {
        NODELET_ERROR_STREAM("Couldn't set exposure mode!");
        return;
    }
}

void RoyaleInRos::callbackMinFiler(const std_msgs::Float32::ConstPtr &msg) {
    m_minFilter = msg->data;
    initMsgUpdate();
}

void RoyaleInRos::callbackMaxFiler(const std_msgs::Float32::ConstPtr &msg) {
    m_maxFilter = msg->data;
    initMsgUpdate();
}

void RoyaleInRos::callbackDivisor(const std_msgs::UInt16::ConstPtr &msg) {
    m_grayDivisor = msg->data;
    initMsgUpdate();
}

void RoyaleInRos::callbackParams(const std_msgs::String::ConstPtr &msg, uint32_t streamIdx) {
    stringstream ss(msg->data);
    string word;
    vector<string> params;
    while (ss >> word) {
        params.push_back(word);
    }
    if (params.size() != 2u) {
        // we should get a parameter name and a value
        NODELET_INFO_STREAM("Invalid parameter : " << msg->data);
        return;
    }

    royale::Variant value;
    royale::VariantType paramType;
    if (getProcessingFlagType(params[0], paramType)) {
        if (paramType == royale::VariantType::Bool) {
            bool bvalue;
            stringstream s2(params[1]);
            s2 >> std::boolalpha >> bvalue;
            value.setBool(bvalue);
        } else if (paramType == royale::VariantType::Int) {
            int32_t ivalue;
            stringstream s2(params[1]);
            s2 >> ivalue;
            value.setInt(static_cast<int>(ivalue));
        } else if (paramType == royale::VariantType::Float) {
            float fvalue;
            stringstream s2(params[1]);
            s2 >> fvalue;
            value.setFloat(fvalue);
        }
    } else {
        NODELET_INFO_STREAM("Processing parameter unknown : " << params[0]);
        return;
    }
    royale::Vector<royale::Pair<royale::String, royale::Variant>> newParam({{params[0], value}});
    auto ret = m_cameraDevice->setProcessingParameters(newParam, m_streamIds[streamIdx]);

    if (ret == CameraStatus::SUCCESS) {
        NODELET_INFO_STREAM("Successfully set parameter : " << m_streamIds[streamIdx] << " " << msg->data);
    } else {
        NODELET_INFO_STREAM("Error setting parameter : " << m_streamIds[streamIdx] << " " << msg->data);
    }
}

PLUGINLIB_EXPORT_CLASS(RoyaleInRos, nodelet::Nodelet)
