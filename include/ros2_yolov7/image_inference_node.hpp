/* NOTE:
 * Large sections of this code are based on code found under the NVIDIA DeepStream
 * Repository here: https://github.com/NVIDIA-AI-IOT/yolo_deepstream
 * The license below is copied as is to meet terms of using that code here.
 */

/* LICENSE:
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/* -------------------------------------------------------------------------- */
/*                                  INCLUDES                                  */
/* -------------------------------------------------------------------------- */

/* ------------------------ Standard Library Includes ----------------------- */
#include <chrono>
#include <memory>
#include <vector>
#include <numeric>
#include <random>
#include <string>

/* ------------------------------ ROS2 Includes ----------------------------- */
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include <message_filters/pass_through.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/latest_time.h>
#include <message_filters/synchronizer.h>

/* ----------------------- TensorRT & YOLOv7 Includes ----------------------- */
#include "Yolov7.h"

/* ----------------------------- OpenCV Includes ---------------------------- */
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv_bridge/cv_bridge.h"

/* -------------------------------------------------------------------------- */
/*                             Class Declarations                             */
/* -------------------------------------------------------------------------- */
class YOLOv7InferenceNode : public rclcpp::Node
{
public:
    // Make an explicit constructor
    explicit YOLOv7InferenceNode();

private:

    /* ------------------ YOLOv7 Private Variable Declarations ------------------ */
    // YoloV7 TensorRT Engine Path
    std::string _engine_path;
    // YoloV7 Detector Object
    std::unique_ptr<Yolov7> _yolov7;

    /* ------------------- Subscriber & Publisher Declarations ------------------ */
    // Image Subscriber
    // TODO: Make this a Synchronized Image Subscriber to get all 6 images at once. 
    message_filters::Subscriber<sensor_msgs::msg::Image> _front_left_image_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> _front_left_center_image_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> _front_right_center_image_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> _front_right_image_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> _rear_right_image_sub;
    message_filters::Subscriber<sensor_msgs::msg::Image> _rear_left_image_sub;

    using SyncPolicy = message_filters::sync_policies::LatestTime<sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image
    >;
    using Sync = message_filters::Synchronizer<SyncPolicy>;
    std::shared_ptr<Sync> _sync_ptr;

    // Camera image with drawn bounding box for debug only
    // TODO: Remove this in production code.
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _camera_img_with_det_pub;

    // Detection Result Publisher.
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr _detection_pub;

    /* ------------------------ Image Handling Variables ------------------------ */
    // Converted CV image ptr
    std::shared_ptr<std::vector<cv::Mat>> _bgr_imgs;
    // CV ptr to msg
    cv_bridge::CvImagePtr _cv_ptr_front_left;
    cv_bridge::CvImagePtr _cv_ptr_front_left_center;
    cv_bridge::CvImagePtr _cv_ptr_front_right_center;
    cv_bridge::CvImagePtr _cv_ptr_front_right;
    cv_bridge::CvImagePtr _cv_ptr_rear_right;
    cv_bridge::CvImagePtr _cv_ptr_rear_left;
    cv_bridge::CvImagePtr _cv_ptr;
    // Output nms results from model
    std::vector<std::vector<std::vector<float>>> _nmsresults;
    int _msg_filter_queue_size = 1;

    /* ------------------------- Image Callback Function ------------------------ */
    void sync_callback(
        const sensor_msgs::msg::Image::ConstSharedPtr & front_left,
        const sensor_msgs::msg::Image::ConstSharedPtr & front_left_center,
        const sensor_msgs::msg::Image::ConstSharedPtr & front_right_center,
        const sensor_msgs::msg::Image::ConstSharedPtr & front_right,
        const sensor_msgs::msg::Image::ConstSharedPtr & rear_right,
        const sensor_msgs::msg::Image::ConstSharedPtr & rear_left
    );

    cv_bridge::CvImagePtr cv_bridge_convert(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
};
