// Copyright (C) 2022 ChenJun
// Copyright (C) 2024 Zheng Yu
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/detector.hpp"


namespace rm_auto_aim
{
static std::string format_id(int color_idx, int cls_idx)
{
  if (color_idx != RED && color_idx != BLUE) {
    return std::string();
  }
  const auto & names = Yolo::classNames();
  if (cls_idx < 0 || cls_idx >= static_cast<int>(names.size())) {
    return std::string();
  }
  const std::string prefix = (color_idx == RED) ? "R_" : "B_";
  return prefix + names[cls_idx];
}

Detector::Detector(
  const int & color, const YoloParams & yolo)
: detect_color(color)
{
  Yolo::Params yolo_params;
  yolo_params.model_path = yolo.model_path;
  yolo_params.score_threshold = yolo.score_threshold;
  yolo_params.nms_threshold = yolo.nms_threshold;
  yolo_params.pre_nms_top_k = yolo.pre_nms_top_k;
  yolo_params.nms_top_k = yolo.nms_top_k;
  yolo_params.anchors = yolo.anchors;
  yolo_params.detect_color = detect_color;
  yolo_ = std::make_unique<Yolo>(yolo_params);
}

std::vector<Armor> Detector::detect(const cv::Mat & input)
{
  armors_.clear();

  if (!yolo_) {
    return armors_;
  }

  yolo_->setDetectColor(detect_color);

  auto dets = yolo_->infer(input);
  armors_.reserve(dets.size());

  for (const auto & det : dets) {
    // det.kpts: TL, BL, BR, TR
    const auto & tl = det.kpts[0];
    const auto & bl = det.kpts[1];
    const auto & br = det.kpts[2];
    const auto & tr = det.kpts[3];

    // Build two "lights" so PnP order stays: LB,LT,RT,RB
    const auto left_box = cv::boundingRect(std::vector<cv::Point2f>{tl, bl});
    const auto right_box = cv::boundingRect(std::vector<cv::Point2f>{tr, br});
    Light left_light(left_box, tl, bl, std::max(1, left_box.area()), 0.0f);
    Light right_light(right_box, tr, br, std::max(1, right_box.area()), 0.0f);

    Armor armor(left_light, right_light);

    const std::string id = format_id(det.color_idx, det.cls_idx);
    if (id.empty()) {
      continue;
    }

    armor.number = id;
    armor.classfication_result = id;
    armor.confidence = det.score;

    // Minimal rule: "1" and "Bb" are large, others are small
    const auto & names = Yolo::classNames();
    const std::string base = (det.cls_idx >= 0 && det.cls_idx < static_cast<int>(names.size())) ?
                               names[det.cls_idx] :
                               std::string();
    armor.type = (base == "1" || base == "Bb") ? ArmorType::LARGE : ArmorType::SMALL;

    armors_.emplace_back(std::move(armor));
  }

  return armors_;
}

void Detector::setYoloThresholds(float score_threshold, float nms_threshold, int nms_top_k)
{
  if (!yolo_) {
    return;
  }
  yolo_->setScoreThreshold(score_threshold);
  yolo_->setNmsThreshold(nms_threshold);
  yolo_->setNmsTopK(nms_top_k);
}

void Detector::setYoloPreNmsTopK(int pre_nms_top_k)
{
  if (!yolo_) {
    return;
  }
  yolo_->setPreNmsTopK(pre_nms_top_k);
}

void Detector::setYoloAnchors(const std::vector<float> & anchors)
{
  if (!yolo_) {
    return;
  }
  yolo_->setAnchors(anchors);
}

Yolo::Timings Detector::lastYoloTimings() const
{
  if (yolo_) {
    return yolo_->lastTimings();
  }
  return Yolo::Timings{};
}

void Detector::drawResults(cv::Mat & img)
{
  for (const auto & armor : armors_) {
    const auto tl = armor.left_light.top;
    const auto bl = armor.left_light.bottom;
    const auto br = armor.right_light.bottom;
    const auto tr = armor.right_light.top;

    const std::vector<cv::Point> poly = {
      cv::Point(static_cast<int>(tl.x), static_cast<int>(tl.y)),
      cv::Point(static_cast<int>(bl.x), static_cast<int>(bl.y)),
      cv::Point(static_cast<int>(br.x), static_cast<int>(br.y)),
      cv::Point(static_cast<int>(tr.x), static_cast<int>(tr.y))};

    cv::polylines(img, poly, true, cv::Scalar(0, 255, 0), 2);

    cv::putText(
      img, armor.classfication_result, tl, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255),
      2);
  }
}

}  // namespace rm_auto_aim
