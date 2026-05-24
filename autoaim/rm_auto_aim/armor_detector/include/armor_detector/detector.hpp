// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/yolo.hpp"

namespace rm_auto_aim
{
class Detector
{
public:
  struct YoloParams
  {
    std::string model_path;
    float score_threshold;
    float nms_threshold;
    int pre_nms_top_k = 500;
    int nms_top_k;
    std::vector<float> anchors;
  };

  Detector(const int & color, const YoloParams & yolo);

  std::vector<Armor> detect(const cv::Mat & input);
  void setYoloThresholds(float score_threshold, float nms_threshold, int nms_top_k);
  void setYoloPreNmsTopK(int pre_nms_top_k);
  void setYoloAnchors(const std::vector<float> & anchors);

  Yolo::Timings lastYoloTimings() const;

  // For debug usage
  void drawResults(cv::Mat & img);

  int detect_color;

private:
  std::vector<Armor> armors_;

  std::unique_ptr<Yolo> yolo_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
