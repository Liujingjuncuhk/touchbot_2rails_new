/****************************************************************************\
 * Copyright (C) 2023 pmdtechnologies ag
 *
 * THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
 * KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
 * PARTICULAR PURPOSE.
 *
 \****************************************************************************/

#include <stdio.h>

#include <QHBoxLayout>
#include <QScrollArea>
#include <QString>
#include <QVBoxLayout>

#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/UInt16.h>

#include "RoyaleControl.hpp"

namespace royale_in_ros {
RoyaleControl::RoyaleControl(QWidget *parent) : rviz::Panel(parent),
                                                m_isInit(false) {

    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        m_isAutomatic[i] = false;
    }

    QVBoxLayout *panelLayout = new QVBoxLayout;

    auto scrollArea = new QScrollArea();
    panelLayout->addWidget(scrollArea);
    setLayout(panelLayout);

    auto panelWidget = new QWidget();
    scrollArea->setWidget(panelWidget);

    // Set UI
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->setSizeConstraint(QLayout::SetMinimumSize);

    ros::master::V_TopicInfo topicInfos;
    ros::master::getTopics(topicInfos);

    mainLayout->addWidget(new QLabel("Camera:"));
    m_cams = new QComboBox;

    std::set<std::string> cameras;

    for (auto curTopic : topicInfos) {
        if (curTopic.name.find("royale_cam_") != std::string::npos) {
            auto div = curTopic.name.find("/", 1);
            if (div != std::string::npos) {
                auto camName = curTopic.name.substr(1, div - 1);
                cameras.insert(camName);
            }
        }
    }

    for (auto curCam : cameras) {
        m_cams->addItem(curCam.c_str());
    }

    mainLayout->addWidget(m_cams);

    // Use Case
    mainLayout->addWidget(new QLabel("Use Case:"));
    m_comboBoxUseCases = new QComboBox;
    mainLayout->addWidget(m_comboBoxUseCases);

    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        mainLayout->addWidget(new QLabel("Stream " + QString::number(i)));

        // Exposure Time
        m_labelExpoTime[i] = new QLabel;
        mainLayout->addWidget(m_labelExpoTime[i]);
        QHBoxLayout *exTimeLayout = new QHBoxLayout;
        exTimeLayout->setSizeConstraint(QLayout::SetMinimumSize);

        m_sliderExpoTime[i] = new QSlider(Qt::Horizontal);
        m_sliderExpoTime[i]->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
        exTimeLayout->addWidget(m_sliderExpoTime[i]);

        m_lineEditExpoTime[i] = new QLineEdit;
        m_lineEditExpoTime[i]->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
        exTimeLayout->addWidget(m_lineEditExpoTime[i]);

        mainLayout->addLayout(exTimeLayout);

        // Auto Exposure
        mainLayout->addWidget(new QLabel("Auto Exposure:"));
        m_checkBoxAutoExpo[i] = new QCheckBox;
        mainLayout->addWidget(m_checkBoxAutoExpo[i]);

        m_labelEditFps[i] = new QLabel;
        mainLayout->addWidget(m_labelEditFps[i]);

        // Auto Exposure
        mainLayout->addWidget(new QLabel("Parameter:"));
        m_lineEditParams[i] = new QLineEdit;
        mainLayout->addWidget(m_lineEditParams[i]);
    }

    // Filter Minimum
    QString minF = QString::number((float)MIN_FILTER / 100.0f, 10, 2);
    QString maxF = QString::number((float)MAX_FILTER / 100.0f, 10, 2);

    mainLayout->addWidget(new QLabel("Filter Minimum (" + minF + " - " + maxF + "m):"));
    QHBoxLayout *minFilterLayout = new QHBoxLayout;

    m_sliderMinFilter = new QSlider(Qt::Horizontal);
    m_sliderMinFilter->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
    m_sliderMinFilter->setRange(MIN_FILTER, MAX_FILTER);
    minFilterLayout->addWidget(m_sliderMinFilter);

    m_lineEditMinFilter = new QLineEdit;
    m_lineEditMinFilter->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    minFilterLayout->addWidget(m_lineEditMinFilter);

    mainLayout->addLayout(minFilterLayout);

    // Filter Maximum
    mainLayout->addWidget(new QLabel("Filter Maximum (" + minF + " - " + maxF + "m):"));
    QHBoxLayout *maxFilterLayout = new QHBoxLayout;

    m_sliderMaxFilter = new QSlider(Qt::Horizontal);
    m_sliderMaxFilter->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
    m_sliderMaxFilter->setRange(MIN_FILTER, MAX_FILTER);
    maxFilterLayout->addWidget(m_sliderMaxFilter);

    m_lineEditMaxFilter = new QLineEdit;
    m_lineEditMaxFilter->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    maxFilterLayout->addWidget(m_lineEditMaxFilter);

    mainLayout->addLayout(maxFilterLayout);

    // Gray Divisor
    QString minD = QString::number(MIN_DIVISOR);
    QString maxD = QString::number(MAX_DIVISOR);

    mainLayout->addWidget(new QLabel("Gray Divisor (" + minD + " - " + maxD + "):"));
    QHBoxLayout *divisorLayout = new QHBoxLayout;

    m_sliderDivisor = new QSlider(Qt::Horizontal);
    m_sliderDivisor->setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::Fixed);
    m_sliderDivisor->setRange(MIN_DIVISOR, MAX_DIVISOR);
    divisorLayout->addWidget(m_sliderDivisor);

    m_lineEditDivisor = new QLineEdit;
    m_lineEditDivisor->setSizePolicy(QSizePolicy::Maximum, QSizePolicy::Fixed);
    divisorLayout->addWidget(m_lineEditDivisor);

    mainLayout->addLayout(divisorLayout);

    panelWidget->setLayout(mainLayout);

    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        connect(m_lineEditExpoTime[i], &QLineEdit::editingFinished, this, [this, i](void) { preciseExposureTimeSetting(i); });
        connect(m_lineEditParams[i], &QLineEdit::editingFinished, this, [this, i](void) { setParameter(i); });

        connect(m_sliderExpoTime[i], &QSlider::valueChanged, this, [this, i](int val) { setExposureTime(val, i); });
        connect(m_checkBoxAutoExpo[i], &QCheckBox::toggled, this, [this, i](bool val) { setExposureMode(val, i); });
    }

    connect(m_comboBoxUseCases, SIGNAL(currentTextChanged(const QString)), this, SLOT(setUseCase(const QString)));
    connect(m_sliderMinFilter, SIGNAL(valueChanged(int)), this, SLOT(setMinFilter(int)));
    connect(m_sliderMaxFilter, SIGNAL(valueChanged(int)), this, SLOT(setMaxFilter(int)));
    connect(m_sliderDivisor, SIGNAL(valueChanged(int)), this, SLOT(setDivisor(int)));

    connect(m_lineEditMinFilter, SIGNAL(editingFinished()), this, SLOT(preciseMinFilterSetting()));
    connect(m_lineEditMaxFilter, SIGNAL(editingFinished()), this, SLOT(preciseMaxFilterSetting()));
    connect(m_lineEditDivisor, SIGNAL(editingFinished()), this, SLOT(preciseDivisorSetting()));

    connect(m_cams, SIGNAL(currentIndexChanged(int)), this, SLOT(chooseCamera(int)));

    m_cams->currentIndexChanged(0);
}

void RoyaleControl::chooseCamera(int idx) {
    std::string curCam = m_cams->itemText(idx).toStdString();

    // Advertise the topics to publish the message of the changed settings in the UI
    m_pubIsInit = m_nh.advertise<std_msgs::Bool>(curCam + "/is_init", 1);
    m_pubUseCase = m_nh.advertise<std_msgs::String>(curCam + "/use_case", 1);
    m_pubMinFilter = m_nh.advertise<std_msgs::Float32>(curCam + "/min_filter", 1);
    m_pubMaxFilter = m_nh.advertise<std_msgs::Float32>(curCam + "/max_filter", 1);
    m_pubDivisor = m_nh.advertise<std_msgs::UInt16>(curCam + "/divisor", 1);
    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        m_pubExpoTime[i] = m_nh.advertise<std_msgs::UInt32>(std::string(curCam + "/expo_time_") + std::to_string(i), 1);
        m_pubExpoMode[i] = m_nh.advertise<std_msgs::Bool>(std::string(curCam + "/expo_mode_") + std::to_string(i), 1);
        m_pubParams[i] = m_nh.advertise<std_msgs::String>(std::string(curCam + "/params_") + std::to_string(i), 1);
    }

    // Subscribe the messages to show the changed state of the camera
    m_subInit = m_nh.subscribe(curCam + "/init_panel", 1, &RoyaleControl::callbackInit, this);

    for (auto i = 0u; i < ROYALE_ROS_MAX_STREAMS; ++i) {
        m_subExpoTimeParam[i] = m_nh.subscribe<std_msgs::String::ConstPtr, const std_msgs::String::ConstPtr &>(curCam + "/expo_time_param_" + std::to_string(i), 1, [this, i](const auto &msg) { callbackExpoTimeParam(msg, i); });
        m_subExpoTimeValue[i] = m_nh.subscribe<std_msgs::UInt32::ConstPtr, const std_msgs::UInt32::ConstPtr &>(curCam + "/expo_time_value_" + std::to_string(i), 1, [this, i](const auto &msg) { callbackExpoTimeValue(msg, i); });
        m_subFps[i] = m_nh.subscribe<std_msgs::String::ConstPtr, const std_msgs::String::ConstPtr &>(curCam + "/update_fps_" + std::to_string(i), 1, [this, i](const auto &msg) { callbackFps(msg, i); });
    }

    // Publish the message to indicate that the initialization is complete
    m_isInit = false;
    std_msgs::Bool msg;
    msg.data = m_isInit;
    m_pubIsInit.publish(msg);
}

void RoyaleControl::callbackInit(const std_msgs::String::ConstPtr &msg) {
    // Will be called (only once) whenever the UI is started
    if (!m_isInit) {
        QString str = msg->data.c_str();
        QStringList initList = str.split('/');

        m_currentUseCase = initList.at(0);

        float min = initList.at(1).toFloat();
        m_sliderMinFilter->blockSignals(true);
        m_sliderMinFilter->setValue((int)(min * 100));
        m_sliderMinFilter->blockSignals(false);
        m_lineEditMinFilter->setText(QString::number(min, 10, 2));

        float max = initList.at(2).toFloat();
        m_sliderMaxFilter->blockSignals(true);
        m_sliderMaxFilter->setValue((int)(max * 100));
        m_sliderMaxFilter->blockSignals(false);
        m_lineEditMaxFilter->setText(QString::number(max, 10, 2));

        m_sliderDivisor->blockSignals(true);
        m_sliderDivisor->setValue(initList.at(3).toInt());
        m_sliderDivisor->blockSignals(false);
        m_lineEditDivisor->setText(initList.at(3));

        m_comboBoxUseCases->blockSignals(true);
        m_comboBoxUseCases->clear();
        for (int i = 4; i < initList.size(); ++i) {
            m_comboBoxUseCases->addItem(initList.at(i));
        }
        auto cmbIdx = m_comboBoxUseCases->findText(m_currentUseCase);
        if (cmbIdx != -1) {
            m_comboBoxUseCases->setCurrentIndex(cmbIdx);
        }
        m_comboBoxUseCases->blockSignals(false);

        // Publish the message to indicate that the initialization is complete
        m_isInit = true;
        std_msgs::Bool msg;
        msg.data = m_isInit;
        m_pubIsInit.publish(msg);
    }
}

void RoyaleControl::callbackExpoTimeParam(const std_msgs::String::ConstPtr &msg, uint32_t streamIdx) {
    QString str = msg->data.c_str();
    m_minETSlider[streamIdx] = str.section('/', 0, 0).toInt();
    m_maxETSlider[streamIdx] = str.section('/', 1, 1).toInt();
    m_sliderExpoTime[streamIdx]->blockSignals(true);
    m_sliderExpoTime[streamIdx]->setRange(m_minETSlider[streamIdx], m_maxETSlider[streamIdx]);
    m_sliderExpoTime[streamIdx]->blockSignals(false);

    m_labelExpoTime[streamIdx]->setText("Exposure Time (" + QString::number(m_minETSlider[streamIdx]) + " - " + QString::number(m_maxETSlider[streamIdx]) + "):");

    m_isAutomatic[streamIdx] = str.section('/', 2, 2).toInt();
    m_checkBoxAutoExpo[streamIdx]->blockSignals(true);
    m_checkBoxAutoExpo[streamIdx]->setChecked(m_isAutomatic[streamIdx]);
    m_checkBoxAutoExpo[streamIdx]->blockSignals(false);
    m_lineEditExpoTime[streamIdx]->setEnabled(!m_isAutomatic[streamIdx]);
}

void RoyaleControl::callbackExpoTimeValue(const std_msgs::UInt32::ConstPtr &msg, uint32_t streamIdx) {
    int value = static_cast<int>(msg->data);
    m_sliderExpoTime[streamIdx]->blockSignals(true);
    m_sliderExpoTime[streamIdx]->setValue(value);
    m_sliderExpoTime[streamIdx]->blockSignals(false);

    m_exposureTime[streamIdx] = value;
    m_lineEditExpoTime[streamIdx]->setText(QString::number(m_exposureTime[streamIdx]));
}

void RoyaleControl::callbackFps(const std_msgs::String::ConstPtr &msg, uint32_t streamIdx) {
    QString fps = msg->data.c_str();
    m_labelEditFps[streamIdx]->setText("Frames per Second: " + fps);
}

void RoyaleControl::setUseCase(const QString &currentMode) {
    std_msgs::String msg;
    msg.data = currentMode.toStdString();
    m_pubUseCase.publish(msg);
}

void RoyaleControl::setExposureTime(int value, uint32_t streamIdx) {
    if (m_isAutomatic[streamIdx]) {
        // The exposure time can not be manually changed in the auto exposure mode
        m_sliderExpoTime[streamIdx]->blockSignals(true);
        m_sliderExpoTime[streamIdx]->setValue(m_exposureTime[streamIdx]);
        m_sliderExpoTime[streamIdx]->blockSignals(false);
    } else {
        m_sliderExpoTime[streamIdx]->blockSignals(true);
        m_sliderExpoTime[streamIdx]->setValue(value);

        // Publish the message to change the exposure time
        std_msgs::UInt32 msg;
        msg.data = static_cast<uint32_t>(value);
        m_pubExpoTime[streamIdx].publish(msg);

        m_exposureTime[streamIdx] = value;
        m_lineEditExpoTime[streamIdx]->setText(QString::number(m_exposureTime[streamIdx]));

        m_sliderExpoTime[streamIdx]->blockSignals(false);
    }
}

void RoyaleControl::setExposureMode(bool isAutomatic, uint32_t streamIdx) {
    // Publish the message to change the exposure mode
    m_isAutomatic[streamIdx] = isAutomatic;
    std_msgs::Bool msg;
    msg.data = isAutomatic;
    m_pubExpoMode[streamIdx].publish(msg);

    // The exposure time can not be manually changed in the auto exposure mode
    m_lineEditExpoTime[streamIdx]->setEnabled(!isAutomatic);
}

void RoyaleControl::setMinFilter(int value) {
    // The minimum value should not be more than the maximum value
    int maxMinFilter = MAX_FILTER - 1;
    if (value <= maxMinFilter) {
        m_sliderMinFilter->blockSignals(true);
        m_sliderMinFilter->setValue(value);

        // Publish the message to change the min filter
        float min = (float)value / 100.0f;
        std_msgs::Float32 msg;
        msg.data = min;
        m_pubMinFilter.publish(msg);

        m_lineEditMinFilter->setText(QString::number(min, 10, 2));

        if (m_sliderMaxFilter->value() < value) {
            setMaxFilter(value + 1);
        }
        m_sliderMinFilter->blockSignals(false);
    } else {
        setMinFilter(maxMinFilter);
    }
}

void RoyaleControl::setMaxFilter(int value) {
    // The maximum value should not be less than the minimum value
    int minMaxFilter = MIN_FILTER + 1;
    if (value >= minMaxFilter) {
        m_sliderMaxFilter->blockSignals(true);
        m_sliderMaxFilter->setValue(value);

        // Publish the message to change the max filter
        float max = (float)value / 100.0f;
        std_msgs::Float32 msg;
        msg.data = max;
        m_pubMaxFilter.publish(msg);

        m_lineEditMaxFilter->setText(QString::number(max, 10, 2));

        if (m_sliderMinFilter->value() > value) {
            setMinFilter(value - 1);
        }
        m_sliderMaxFilter->blockSignals(false);
    } else {
        setMaxFilter(minMaxFilter);
    }
}

void RoyaleControl::setDivisor(int value) {
    m_sliderDivisor->blockSignals(true);
    m_sliderDivisor->setValue(value);

    // Publish the message to change the gray divisor
    std_msgs::UInt16 msg;
    msg.data = static_cast<uint16_t>(value);
    m_pubDivisor.publish(msg);

    m_lineEditDivisor->setText(QString::number(value));
    m_sliderDivisor->blockSignals(false);
}

void RoyaleControl::preciseExposureTimeSetting(uint32_t streamIdx) {
    int value = m_lineEditExpoTime[streamIdx]->text().toInt();
    // The input is confirmed by the Enter key, otherwise it has not changed
    // The input which is out of range input is not allowed
    if (value >= m_minETSlider[streamIdx] && value <= m_maxETSlider[streamIdx] && m_lineEditExpoTime[streamIdx]->hasFocus()) {
        setExposureTime(value, streamIdx);
    } else {
        m_lineEditExpoTime[streamIdx]->setText(QString::number(m_exposureTime[streamIdx]));
    }
}

void RoyaleControl::preciseMinFilterSetting() {
    int value = (int)(m_lineEditMinFilter->text().toFloat() * 100);
    // The input is confirmed by the Enter key, otherwise it has not changed
    // The input which is out of range input is not allowed
    if (value >= MIN_FILTER && m_lineEditMinFilter->hasFocus()) {
        setMinFilter(value);
    } else {
        float min = (float)m_sliderMinFilter->value() / 100.0f;
        m_lineEditMinFilter->setText(QString::number(min, 10, 2));
    }
}

void RoyaleControl::preciseMaxFilterSetting() {
    int value = (int)(m_lineEditMaxFilter->text().toFloat() * 100);
    // The input is confirmed by the Enter key, otherwise it has not changed
    // The input which is out of range input is not allowed
    if (value <= MAX_FILTER && m_lineEditMaxFilter->hasFocus()) {
        setMaxFilter(value);
    } else {
        float max = (float)m_sliderMaxFilter->value() / 100.0f;
        m_lineEditMaxFilter->setText(QString::number(max, 10, 2));
    }
}

void RoyaleControl::preciseDivisorSetting() {
    int value = m_lineEditDivisor->text().toInt();
    // The input is confirmed by the Enter key, otherwise it has not changed
    // The input which is out of range input is not allowed
    if (value >= MIN_DIVISOR && value <= MAX_DIVISOR && m_lineEditDivisor->hasFocus()) {
        setDivisor(value);
    } else {
        m_lineEditDivisor->setText(QString::number(m_sliderDivisor->value()));
    }
}

void RoyaleControl::setParameter(uint32_t streamIdx) {
    if (!m_lineEditParams[streamIdx]->text().isEmpty()) {
        std_msgs::String msg;
        msg.data = m_lineEditParams[streamIdx]->text().toStdString();
        m_lineEditParams[streamIdx]->clear();
        m_pubParams[streamIdx].publish(msg);
    }
}

} // namespace royale_in_ros

PLUGINLIB_EXPORT_CLASS(royale_in_ros::RoyaleControl, rviz::Panel)
