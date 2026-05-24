// Copyright (C) 2026
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__YOLO_HPP_
#define ARMOR_DETECTOR__YOLO_HPP_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rm_auto_aim
{

class Yolo
{
public:
  struct Params
  {
    std::string model_path;
    float score_threshold = 0.65f;
    float nms_threshold = 0.45f;
    int pre_nms_top_k = 500;
    int nms_top_k = 300;
    // 18 floats: s(3*2), m(3*2), l(3*2)
    std::vector<float> anchors;

    // detect_color: -1 keep red+blue; 0 keep red; 1 keep blue
    int detect_color = -1;
  };

  struct Detection
  {
    // Keypoints in original image coordinates, order: TL, BL, BR, TR
    std::array<cv::Point2f, 4> kpts;
    int color_idx = -1;  // 0 red, 1 blue, 2 gray, 3 purple
    int cls_idx = -1;    // 0..8 => {G,1,2,3,4,5,O,Bs,Bb}
    float obj = 0.0f;
    float cls_score = 0.0f;
    float score = 0.0f;  // obj * cls_score
  };

  struct Timings
  {
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    int src_w = 0;
    int src_h = 0;
    int input_w = 0;
    int input_h = 0;
    bool letterbox_used = false;
    float scale = 1.0f;
    int x_shift = 0;
    int y_shift = 0;
  };

  explicit Yolo(const Params & params);
  ~Yolo();

  Yolo(const Yolo &) = delete;
  Yolo & operator=(const Yolo &) = delete;

  std::vector<Detection> infer(const cv::Mat & bgr);
  void setDetectColor(int detect_color);
  void setScoreThreshold(float score_threshold);
  void setNmsThreshold(float nms_threshold);
  void setPreNmsTopK(int pre_nms_top_k);
  void setNmsTopK(int nms_top_k);
  void setAnchors(const std::vector<float> & anchors);

  const Timings & lastTimings() const;

  static const std::vector<std::string> & classNames();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__YOLO_HPP_
