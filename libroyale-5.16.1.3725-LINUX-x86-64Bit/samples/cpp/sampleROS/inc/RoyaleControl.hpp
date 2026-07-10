/****************************************************************************\
 * Copyright (C) 2023 pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <rviz/panel.h>

#include <std_msgs/String.h>
#include <std_msgs/UInt32.h>

#include <QCheckBox>
#include <QComboBox>
#include <QLabel>
#include <QLineEdit>
#include <QSlider>

#define MIN_FILTER 0
#define MAX_FILTER 750
#define MIN_DIVISOR 1
#define MAX_DIVISOR 2000

#define ROYALE_ROS_MAX_STREAMS 2u

namespace royale_in_ros {
class RoyaleControl : public rviz::Panel {
    Q_OBJECT

  public:
    RoyaleControl(QWidget *parent = 0);

  private Q_SLOTS:
    void setUseCase(const QString &currentMode);
    void setExposureTime(int value, uint32_t streamIdx);
    void setExposureMode(bool isAutomatic, uint32_t streamIdx);
    void setMinFilter(int value);
    void setMaxFilter(int value);
    void setDivisor(int value);

    // The precise value can be entered directly via the text editor.
    void preciseExposureTimeSetting(uint32_t streamIdx);
    void preciseMinFilterSetting();
    void preciseMaxFilterSetting();
    void preciseDivisorSetting();
    void setParameter(uint32_t streamIdx);

    void chooseCamera(int idx);

  private:
    // Callback the camera initial settings after the UI is started
    void callbackInit(const std_msgs::String::ConstPtr &msg);

    // Callback the current parameters of exposure time
    // when the UI is started or user case is switched
    void callbackExpoTimeParam(const std_msgs::String::ConstPtr &msg, uint32_t streamIdx);
    void callbackExpoTimeValue(const std_msgs::UInt32::ConstPtr &msg, uint32_t streamIdx);

    // Callback the fps and display it
    void callbackFps(const std_msgs::String::ConstPtr &msg, uint32_t streamIdx);

    QComboBox *m_cams;
    QComboBox *m_comboBoxUseCases;
    QLabel *m_labelExpoTime[ROYALE_ROS_MAX_STREAMS];
    QSlider *m_sliderExpoTime[ROYALE_ROS_MAX_STREAMS];
    QLineEdit *m_lineEditExpoTime[ROYALE_ROS_MAX_STREAMS];
    QCheckBox *m_checkBoxAutoExpo[ROYALE_ROS_MAX_STREAMS];
    QLabel *m_labelEditFps[ROYALE_ROS_MAX_STREAMS];
    QSlider *m_sliderDivisor;
    QLineEdit *m_lineEditDivisor;
    QSlider *m_sliderMinFilter;
    QLineEdit *m_lineEditMinFilter;
    QSlider *m_sliderMaxFilter;
    QLineEdit *m_lineEditMaxFilter;
    QLineEdit *m_lineEditParams[ROYALE_ROS_MAX_STREAMS];
    QString m_currentUseCase;

    ros::NodeHandle m_nh;
    ros::Publisher m_pubIsInit;
    ros::Publisher m_pubUseCase;
    ros::Publisher m_pubExpoTime[ROYALE_ROS_MAX_STREAMS];
    ros::Publisher m_pubExpoMode[ROYALE_ROS_MAX_STREAMS];
    ros::Publisher m_pubMinFilter;
    ros::Publisher m_pubMaxFilter;
    ros::Publisher m_pubDivisor;
    ros::Publisher m_pubParams[ROYALE_ROS_MAX_STREAMS];

    ros::Subscriber m_subInit;
    ros::Subscriber m_subExpoTimeParam[ROYALE_ROS_MAX_STREAMS];
    ros::Subscriber m_subExpoTimeValue[ROYALE_ROS_MAX_STREAMS];
    ros::Subscriber m_subFps[ROYALE_ROS_MAX_STREAMS];

    bool m_isInit;
    bool m_isAutomatic[ROYALE_ROS_MAX_STREAMS];
    int m_exposureTime[ROYALE_ROS_MAX_STREAMS];
    int m_minETSlider[ROYALE_ROS_MAX_STREAMS];
    int m_maxETSlider[ROYALE_ROS_MAX_STREAMS];
};
} // namespace royale_in_ros
