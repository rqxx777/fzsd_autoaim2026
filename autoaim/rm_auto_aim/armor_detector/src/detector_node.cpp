// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rcutils/logging.h>
#include <rclcpp/duration.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/detector_node.hpp"

namespace rm_auto_aim
{
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: Node("armor_detector", options)
{
  RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

  // Topic params (keep compatibility with existing bringup yamls)
  const std::string camera_info_topic =
    this->declare_parameter("camera_info_topic", std::string("/camera_info"));
  const std::string image_topic = this->declare_parameter("image_topic", std::string("/image_raw"));
  use_hik_sdk_ = this->declare_parameter("input.use_hik_sdk", true);
  direct_frame_id_ = this->declare_parameter("input.frame_id", std::string("camera_optical_frame"));

  // Detector
  detector_ = initDetector();

  // Armors Publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
    "/detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker Publisher
  // See http://wiki.ros.org/rviz/DisplayTypes/Marker
  armor_marker_.ns = "armors";
  armor_marker_.action = visualization_msgs::msg::Marker::ADD;
  armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
  armor_marker_.scale.x = 0.05;
  armor_marker_.scale.z = 0.125;
  armor_marker_.color.a = 1.0;
  armor_marker_.color.g = 0.5;
  armor_marker_.color.b = 1.0;
  armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

  marker_pub_ =
    this->create_publisher<visualization_msgs::msg::MarkerArray>("/detector/marker", 10);

  // Debug Publishers
  debug_ = this->declare_parameter("debug", false);
  debug_timing_every_n_ = this->declare_parameter("debug_timing_every_n", 20);
  if (debug_) {
    createDebugPublishers();
  }

  // Task subscriber
  is_aim_task_ = true;
  task_sub_ = this->create_subscription<std_msgs::msg::String>(
    "/task_mode", 10, std::bind(&ArmorDetectorNode::taskCallback, this, std::placeholders::_1));

  // Debug param change moniter
  debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  debug_cb_handle_ =
    debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter & p) {
      debug_ = p.as_bool();
      debug_ ? createDebugPublishers() : destroyDebugPublishers();
    });

  if (use_hik_sdk_) {
    initPnpSolverFromParams();
    startHikInput();
  } else {
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic, rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
        cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
        cam_info_sub_.reset();
      });

    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS().keep_last(1),
      std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));
  }

  infer_running_.store(true);
  infer_thread_ = std::thread(&ArmorDetectorNode::inferenceLoop, this);
}

ArmorDetectorNode::~ArmorDetectorNode()
{
  stopHikInput();
  infer_running_.store(false);
  infer_cv_.notify_all();
  if (infer_thread_.joinable()) {
    infer_thread_.join();
  }
}

void ArmorDetectorNode::initPnpSolverFromParams()
{
  const auto camera_matrix_v = this->declare_parameter("camera_matrix", std::vector<double>{});
  const auto dist_coeffs_v = this->declare_parameter("distortion_coefficients", std::vector<double>{});

  if (camera_matrix_v.size() != 9 || dist_coeffs_v.size() < 5) {
    throw std::runtime_error(
      "input.use_hik_sdk=true requires camera_matrix(9) and distortion_coefficients(>=5)");
  }

  std::array<double, 9> camera_matrix{};
  for (size_t i = 0; i < 9; ++i) {
    camera_matrix[i] = camera_matrix_v[i];
  }
  std::vector<double> dist_coeffs(dist_coeffs_v.begin(), dist_coeffs_v.begin() + 5);

  cam_center_ = cv::Point2f(static_cast<float>(camera_matrix[2]), static_cast<float>(camera_matrix[5]));
  pnp_solver_ = std::make_unique<PnPSolver>(camera_matrix, dist_coeffs);
}

void ArmorDetectorNode::enqueueImage(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg)
{
  {
    std::lock_guard<std::mutex> lock(infer_mutex_);
    pending_img_msg_ = img_msg;
    has_pending_img_ = true;
  }
  infer_cv_.notify_one();
}

void ArmorDetectorNode::taskCallback(const std_msgs::msg::String::SharedPtr task_msg)
{
  std::string task_mode = task_msg->data;
  if (task_mode == "aim") {
    is_aim_task_ = true;
  } else {
    is_aim_task_ = false;
  }
}

void ArmorDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr img_msg)
{
  enqueueImage(img_msg);
}

void ArmorDetectorNode::startHikInput()
{
  hik_exposure_time_ = this->declare_parameter("hik.exposure_time", 3000.0);
  hik_gain_ = this->declare_parameter("hik.gain", 10.0);
  hik_get_timeout_ms_ = this->declare_parameter("hik.get_timeout_ms", 1000);

  std::string err;
  if (!hik_camera_source_.open(hik_exposure_time_, hik_gain_, hik_get_timeout_ms_, &err)) {
    throw std::runtime_error("Failed to open Hik camera source: " + err);
  }

  capture_running_.store(true);
  capture_thread_ = std::thread(&ArmorDetectorNode::captureLoop, this);
}

void ArmorDetectorNode::stopHikInput()
{
  capture_running_.store(false);
  if (capture_thread_.joinable()) {
    capture_thread_.join();
  }
  hik_camera_source_.close();
}

void ArmorDetectorNode::captureLoop()
{
  while (rclcpp::ok() && capture_running_.load()) {
    hik_camera::HikFrame frame;
    std::string err;
    if (!hik_camera_source_.read(frame, &err)) {
      continue;
    }

    const auto width = static_cast<size_t>(frame.bgr.cols);
    const auto height = static_cast<size_t>(frame.bgr.rows);
    if (width == 0 || height == 0) {
      continue;
    }

    auto img_msg = std::make_shared<sensor_msgs::msg::Image>();
    // Use ROS clock here so detector/tracker timestamps are in the same time domain as TF cache.
    img_msg->header.stamp = this->now();
    img_msg->header.frame_id = direct_frame_id_;
    img_msg->height = static_cast<uint32_t>(height);
    img_msg->width = static_cast<uint32_t>(width);
    img_msg->encoding = "bgr8";
    img_msg->is_bigendian = false;
    img_msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(width * 3);
    img_msg->data.assign(frame.bgr.datastart, frame.bgr.dataend);
    enqueueImage(img_msg);
  }
}

void ArmorDetectorNode::inferenceLoop()
{
  while (rclcpp::ok()) {
    sensor_msgs::msg::Image::ConstSharedPtr img_msg;
    {
      std::unique_lock<std::mutex> lock(infer_mutex_);
      infer_cv_.wait(lock, [this]() {
        return !infer_running_.load() || has_pending_img_;
      });

      if (!infer_running_.load() && !has_pending_img_) {
        return;
      }

      img_msg = pending_img_msg_;
      has_pending_img_ = false;
    }

    processImage(img_msg);
  }
}

void ArmorDetectorNode::processImage(const sensor_msgs::msg::Image::ConstSharedPtr & img_msg)
{
  if (!img_msg) {
    return;
  }

  auto armors = detectArmors(img_msg);

  if (pnp_solver_ != nullptr && is_aim_task_) {
    armors_msg_.header = armor_marker_.header = text_marker_.header = img_msg->header;
    armors_msg_.armors.clear();
    marker_array_.markers.clear();
    armor_marker_.id = 0;
    text_marker_.id = 0;

    auto_aim_interfaces::msg::Armor armor_msg;
    for (const auto & armor : armors) {
      cv::Mat rvec, tvec;
      bool success = pnp_solver_->solvePnP(armor, rvec, tvec);
      if (success) {
        // Fill basic info
        armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armor.type)];
        armor_msg.number = armor.number;

        // Fill pose
        armor_msg.pose.position.x = tvec.at<double>(0);
        armor_msg.pose.position.y = tvec.at<double>(1);
        armor_msg.pose.position.z = tvec.at<double>(2);
        // rvec to 3x3 rotation matrix
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        // rotation matrix to quaternion
        tf2::Matrix3x3 tf2_rotation_matrix(
          rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
          rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
          rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
          rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
          rotation_matrix.at<double>(2, 2));
        tf2::Quaternion tf2_q;
        tf2_rotation_matrix.getRotation(tf2_q);
        armor_msg.pose.orientation = tf2::toMsg(tf2_q);

        // Fill the distance to image center
        armor_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);

        // Fill keypoints
        armor_msg.kpts.clear();
        for (const auto & pt :
             {armor.left_light.top, armor.left_light.bottom, armor.right_light.bottom,
              armor.right_light.top}) {
          geometry_msgs::msg::Point point;
          point.x = pt.x;
          point.y = pt.y;
          armor_msg.kpts.emplace_back(point);
        }

        // Fill the markers
        armor_marker_.id++;
        armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
        armor_marker_.pose = armor_msg.pose;
        text_marker_.id++;
        text_marker_.pose.position = armor_msg.pose.position;
        text_marker_.pose.position.y -= 0.1;
        text_marker_.text = armor.classfication_result;
        armors_msg_.armors.emplace_back(armor_msg);
        marker_array_.markers.emplace_back(armor_marker_);
        marker_array_.markers.emplace_back(text_marker_);
      } else {
        RCLCPP_WARN(this->get_logger(), "PnP failed!");
      }
    }

    // Publishing detected armors
    armors_pub_->publish(armors_msg_);

    // Publishing marker
    publishMarkers();
  }
}

std::unique_ptr<Detector> ArmorDetectorNode::initDetector()
{
  auto detect_color = declare_parameter("detect_color", RED);

  // YOLO params
  Detector::YoloParams yolo_params;
  yolo_params.model_path = this->declare_parameter("yolo.model_path", std::string(""));
  yolo_params.score_threshold =
    static_cast<float>(this->declare_parameter("yolo.score_threshold", 0.65));
  yolo_params.nms_threshold = static_cast<float>(this->declare_parameter("yolo.nms_threshold", 0.45));
  yolo_params.pre_nms_top_k = this->declare_parameter("yolo.pre_nms_top_k", 500);
  yolo_params.nms_top_k = this->declare_parameter("yolo.nms_top_k", 300);
  yolo_pre_nms_top_k_ = yolo_params.pre_nms_top_k;
  yolo_nms_top_k_ = yolo_params.nms_top_k;
  {
    const std::vector<double> default_anchors = {
      10.0, 13.0, 16.0, 30.0, 33.0, 23.0,
      30.0, 61.0, 62.0, 45.0, 59.0, 119.0,
      116.0, 90.0, 156.0, 198.0, 373.0, 326.0};
    const auto anchors = this->declare_parameter("yolo.anchors", default_anchors);
    yolo_params.anchors.reserve(anchors.size());
    for (const auto & a : anchors) {
      yolo_params.anchors.emplace_back(static_cast<float>(a));
    }
  }

  {
    if (yolo_params.model_path.empty()) {
      RCLCPP_FATAL(this->get_logger(), "Set required parameter: yolo.model_path");
      throw std::runtime_error("YOLO model_path is empty");
    }
    std::ifstream f(yolo_params.model_path);
    if (!f.good()) {
      RCLCPP_FATAL(
        this->get_logger(),
        "YOLO model file not found: '%s'. Set parameter yolo.model_path.",
        yolo_params.model_path.c_str());
      throw std::runtime_error("YOLO model file not found");
    }
  }

  return std::make_unique<Detector>(detect_color, yolo_params);
}

std::vector<Armor> ArmorDetectorNode::detectArmors(
  const sensor_msgs::msg::Image::ConstSharedPtr & img_msg)
{
  using clock = std::chrono::steady_clock;
  static auto last_fps_time = clock::now();
  static uint64_t frame_count_since = 0;

  // Convert ROS img to cv::Mat
  auto img = cv_bridge::toCvShare(img_msg, "bgr8")->image;

  // Update params
  detector_->detect_color = get_parameter("detect_color").as_int();
  detector_->setYoloThresholds(
    static_cast<float>(get_parameter("yolo.score_threshold").as_double()),
    static_cast<float>(get_parameter("yolo.nms_threshold").as_double()),
    yolo_nms_top_k_);
  detector_->setYoloPreNmsTopK(get_parameter("yolo.pre_nms_top_k").as_int());

  auto armors = detector_->detect(img);
  const auto yolo_t = detector_->lastYoloTimings();

  auto final_time = this->now();
  auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
  RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");

  // Publish debug info
  if (debug_) {
    auto t_draw0 = clock::now();
    detector_->drawResults(img);
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(
      img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    auto t_draw1 = clock::now();
    double draw_ms = std::chrono::duration<double, std::milli>(t_draw1 - t_draw0).count();

    result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "bgr8", img).toImageMsg());

    debug_frame_count_++;
    frame_count_since++;
    if (debug_timing_every_n_ > 0 && (debug_frame_count_ % static_cast<uint64_t>(debug_timing_every_n_)) == 0) {
      auto now = clock::now();
      double fps = 0.0;
      auto dt = std::chrono::duration<double>(now - last_fps_time).count();
      if (dt > 0.0) {
        fps = static_cast<double>(frame_count_since) / dt;
      }
      last_fps_time = now;
      frame_count_since = 0;
      RCLCPP_DEBUG(
        this->get_logger(),
        "input[%s %dx%d] model[%dx%d] letterbox=%d scale=%.3f shift=(%d,%d) fps=%.2f timing(ms) pre=%.2f infer=%.2f post=%.2f draw=%.2f",
        img_msg->encoding.c_str(),
        yolo_t.src_w, yolo_t.src_h,
        yolo_t.input_w, yolo_t.input_h,
        yolo_t.letterbox_used ? 1 : 0,
        yolo_t.scale,
        yolo_t.x_shift, yolo_t.y_shift,
        fps,
        yolo_t.preprocess_ms, yolo_t.infer_ms, yolo_t.postprocess_ms, draw_ms);
    }
  }

  return armors;
}

void ArmorDetectorNode::createDebugPublishers()
{
  result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::destroyDebugPublishers()
{
  result_img_pub_.shutdown();
}

void ArmorDetectorNode::publishMarkers()
{
  using Marker = visualization_msgs::msg::Marker;
  armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
  marker_array_.markers.emplace_back(armor_marker_);
  marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)
