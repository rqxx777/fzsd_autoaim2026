// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_NODE_HPP_
#define ARMOR_DETECTOR__DETECTOR_NODE_HPP_

// ROS
#include <geometry_msgs/msg/point.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// STD
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "armor_detector/detector.hpp"
#include "armor_detector/pnp_solver.hpp"
#include "auto_aim_interfaces/msg/armors.hpp"
#include "hik_camera/hik_camera_source.hpp"

namespace rm_auto_aim
{

class ArmorDetectorNode : public rclcpp::Node
{
public:
  ArmorDetectorNode(const rclcpp::NodeOptions & options);
  ~ArmorDetectorNode() override;

private:
  void enqueueImage(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg);
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg);
  void inferenceLoop();
  void processImage(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg);
  void initPnpSolverFromParams();
  void startHikInput();
  void stopHikInput();
  void captureLoop();

  std::unique_ptr<Detector> initDetector();
  std::vector<Armor> detectArmors(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg);

  void createDebugPublishers();
  void destroyDebugPublishers();

  void publishMarkers();

  //  task subscriber
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr task_sub_;
  bool is_aim_task_;
  void taskCallback(const std_msgs::msg::String::SharedPtr task_msg);

  // Armor Detector
  std::unique_ptr<Detector> detector_;

  // Detected armors publisher
  auto_aim_interfaces::msg::Armors armors_msg_;
  rclcpp::Publisher<auto_aim_interfaces::msg::Armors>::SharedPtr armors_pub_;

  // Visualization marker publisher
  visualization_msgs::msg::Marker armor_marker_;
  visualization_msgs::msg::Marker text_marker_;
  visualization_msgs::msg::MarkerArray marker_array_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // Camera info part
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  cv::Point2f cam_center_;
  std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
  std::unique_ptr<PnPSolver> pnp_solver_;

  // Image subscrpition
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;

  // Direct camera input (non-ROS topic)
  bool use_hik_sdk_ = false;
  std::string direct_frame_id_ = "camera_optical_frame";
  double hik_exposure_time_ = 6000.0;
  double hik_gain_ = 10.0;
  int hik_get_timeout_ms_ = 1000;
  hik_camera::HikCameraSource hik_camera_source_;
  std::thread capture_thread_;
  std::atomic<bool> capture_running_{false};

  // Async inference worker (latest frame only)
  std::thread infer_thread_;
  std::mutex infer_mutex_;
  std::condition_variable infer_cv_;
  sensor_msgs::msg::Image::ConstSharedPtr pending_img_msg_;
  bool has_pending_img_ = false;
  std::atomic<bool> infer_running_{false};

  // Debug information
  bool debug_;
  int debug_timing_every_n_;
  uint64_t debug_frame_count_ = 0;
  std::shared_ptr<rclcpp::ParameterEventHandler> debug_param_sub_;
  std::shared_ptr<rclcpp::ParameterCallbackHandle> debug_cb_handle_;
  image_transport::Publisher result_img_pub_;

  int yolo_nms_top_k_ = 300;
  int yolo_pre_nms_top_k_ = 500;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_NODE_HPP_
