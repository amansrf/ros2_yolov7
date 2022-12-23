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
#include "vision_msgs/msg/detection3_d_array.hpp"

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
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _image_sub;

    // Camera image with drawn bounding box for debug only
    // TODO: Remove this in production code.
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _camera_img_with_det_pub;

    // Detection Result Publisher.
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr _detection_pub;

    /* ------------------------ Image Handling Variables ------------------------ */
    // Converted CV image ptr
    std::shared_ptr<std::vector<cv::Mat>> _bgr_imgs;
    // CV ptr to msg
    cv_bridge::CvImagePtr _cv_ptr;
    // Output nms results from model
    std::vector<std::vector<std::vector<float>>> _nmsresults;

    /* ------------------------- Image Callback Function ------------------------ */
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg);
};
