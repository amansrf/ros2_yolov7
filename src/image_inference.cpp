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
#include <sensor_msgs/msg/image.hpp>

/* ----------------------- TensorRT & YOLOv7 Includes ----------------------- */
#include <Yolov7.h>

/* ----------------------------- OpenCV Includes ---------------------------- */
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

/* -------------------------------------------------------------------------- */
/*                           CONSTANTS & PARAMETERS                           */
/* -------------------------------------------------------------------------- */
const std::string ENGINE_PATH = "/home/roar/ART/perception/model_trials/NVIDIA_AI_IOT_tensorrt_yolov7/yolo_deepstream/tensorrt_yolov7/build/yolov7PTQ.engine";

/* -------------------------------------------------------------------------- */
/*                              HELPER FUNCTIONS                              */
/* -------------------------------------------------------------------------- */

// TODO: FIX THIS for a static implementation or dynamic ros2 implementation.
// std::string parse_model_path(argsParser& cmdLine) {
//     const char* engine_path_str = cmdLine.ParseString("engine");
//     std::string engine_path;
//     if (engine_path_str) engine_path = std::string(engine_path_str);
//     return engine_path;
// }

/* -------------------------------------------------------------------------- */
/*                                 ROS2 NODE                                  */
/* -------------------------------------------------------------------------- */
using namespace std::chrono_literals;

class YOLOv7InferenceNode : public rclcpp::Node
{
  public:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    
    /* ---------------------------- YOLOv7 Init Stuff --------------------------- */
    Yolov7 yolov7;
    yolov7 = Yolov7(std::string(ENGINE_PATH));
    std::vector<cv::Mat> bgr_imgs;

    YOLOv7InferenceNode()
    : Node("yolov7_inference_node")
    {
      /* ------------------------------- QOS Profile ------------------------------ */
      rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
      auto sensor_msgs_qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, 1), qos_profile);

      /* --------------------------- Image Subscription --------------------------- */
      image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/vimba_front_left_center/image",
        sensor_msgs_qos,
        std::bind(&YOLOv7InferenceNode::image_callback, this, std::placeholders::_1)
      );

      /* ----------------------------- Image Publisher ---------------------------- */
      image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        "/vimba_front_left_center/det_image",
        sensor_msgs_qos
      );
    }


private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    /* ------------------ Receive msg and convert to cv2 format ----------------- */
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    }
    catch (cv_bridge::Exception& e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    /* -------------------------- Preprocess the images ------------------------- */
    this->bgr_imgs.push_back(cv_ptr->image);
    (this->yolov7).preProcess(this->bgr_imgs);

    /* ------------------------------ Run Inference ----------------------------- */
    yolov7.infer();

    /* -------------------------- Run NMS & PostProcess ------------------------- */
    std::vector<std::vector<std::vector<float>>> nmsresults = yolov7().PostProcess();

    for(int j =0; j < nmsresults.size();j++){
        // TODO: Publish here!
        Yolov7::DrawBoxesonGraph(bgr_imgs[j],nmsresults[j]);

        cv_ptr->image = bgr_imgs[j];
        image_pub_.publish(cv_ptr->toImageMsg());
        // std::string output_path = img_paths[j] + "detect" + std::to_string(j)+".jpg";      
        // cv::imwrite(output_path, bgr_imgs[j]);
        // std::cout<<"detectec image written to: "<<output_path<<std::endl;
    }


  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
