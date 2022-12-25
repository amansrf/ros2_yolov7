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

#include "ros2_yolov7/image_inference_node.hpp"
#include "ros2_yolov7/camera_intrinsics.hpp"

/* -------------------------------------------------------------------------- */
/*                                 ROS2 NODE                                  */
/* -------------------------------------------------------------------------- */
using namespace std::chrono_literals;

YOLOv7InferenceNode::YOLOv7InferenceNode() : rclcpp::Node("yolov7_inference_node")
{
    // TODO: Make _engine_path a launch parameter
    _engine_path = "/home/roar/ART/perception/model_trials/NVIDIA_AI_IOT_tensorrt_yolov7/yolo_deepstream/tensorrt_yolov7/build/yolov7PTQ.engine";

    // Initialize the YOLOv7 Object
    _yolov7 = std::make_unique<Yolov7>(_engine_path);

    // Declare a pointer to bgr msgs in CV2 format
    _bgr_imgs = std::make_shared<std::vector<cv::Mat>>();
    // TODO: Relook at how many images you want to store here when generalizing to multiple cameras.
    _bgr_imgs->reserve(1);


    /* ------------------------------- QOS Profile ------------------------------ */
    rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
    auto sensor_msgs_qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

    // TODO: Make the camera topic variable and do synchronous subscription over 6 topics.
    /* --------------------------- Image Subscription --------------------------- */
    _image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "vimba_front_left_center/image",
        sensor_msgs_qos,
        std::bind(&YOLOv7InferenceNode::image_callback, this, std::placeholders::_1)
    );

    /* ----------------------------- Image Publisher ---------------------------- */
    // TODO: DEBUG ONLY. REMOVE FROM PRODUCTION CODE.
    _camera_img_with_det_pub = this->create_publisher<sensor_msgs::msg::Image>(
        "/vimba_front_left_center/det_image",
        sensor_msgs_qos
    );

    /* --------------------------- Detection Publisher -------------------------- */
    _detection_pub = this->create_publisher<vision_msgs::msg::Detection3DArray>(
        "/vimba_front_left_center/det3d",
        sensor_msgs_qos
    );
}


void YOLOv7InferenceNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    /* ------------------ Receive msg and convert to cv2 format ----------------- */
    try
    {
        _cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Assign the image to memory allocated to it, if unassigned, otherwise overwrite
    if (_bgr_imgs->empty())
    {
        _bgr_imgs->push_back(_cv_ptr->image);
    }
    else
    {
        _bgr_imgs->at(0) = _cv_ptr->image;
    }

    // Preprocess the images
    _yolov7->preProcess(*_bgr_imgs);

    // Run Inference
    _yolov7->infer();

    // Run NMS & PostProcess
    _nmsresults = _yolov7->PostProcess();

    // Initialize a detection 3D array
    vision_msgs::msg::Detection3DArray det3d;
    det3d.header.stamp      = msg->header.stamp;
    det3d.header.frame_id   = "vimba_front_left_center";


    for(int detection_index = 0; detection_index < _nmsresults.size(); detection_index++)
    {
        // TODO: We seem to writing to higher indexes than those that have been allocated.
        Yolov7::DrawBoxesonGraph(_bgr_imgs->at(detection_index), _nmsresults[detection_index]);

        for(int i = 0; i < _nmsresults[detection_index].size(); ++i)
        {
            auto& ibox          = _nmsresults[detection_index][i];
            float left          = ibox[0];
            float top           = ibox[1];
            float right         = ibox[2];
            float bottom        = ibox[3];
            int class_label     = ibox[4];
            float confidence    = ibox[5];

            
        }
        _cv_ptr->image = _bgr_imgs->at(detection_index);
        _camera_img_with_det_pub->publish(*(_cv_ptr->toImageMsg()).get());
    }
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<YOLOv7InferenceNode>());
  rclcpp::shutdown();
  return 0;
}
