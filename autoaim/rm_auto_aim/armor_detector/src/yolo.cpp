// Copyright (C) 2026
// Licensed under the MIT License.

#include "armor_detector/yolo.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/hb_sys.h"

namespace rm_auto_aim
{

static float sigmoid(float x)
{
  return 1.0f / (1.0f + std::exp(-x));
}

static void hbCheck(int ret, const char * what)
{
  if (ret != 0) {
    throw std::runtime_error(std::string(what) + ", ret=" + std::to_string(ret));
  }
}

static std::vector<std::string> buildClassNames()
{
  return {"G", "1", "2", "3", "4", "5", "O", "Bs", "Bb"};
}

const std::vector<std::string> & Yolo::classNames()
{
  static const std::vector<std::string> kNames = buildClassNames();
  return kNames;
}

struct Yolo::Impl
{
  static constexpr int kFeatureDim = 22;
  static constexpr int kColorClassCount = 4;
  static constexpr int kClassClassCount = 9;

  Params params;
  Timings last_timings;
  std::mutex timings_mutex;

  hbPackedDNNHandle_t packed = nullptr;
  hbDNNHandle_t model = nullptr;

  hbDNNTensor input;
  std::vector<hbDNNTensor> outputs;
  hbDNNTensorProperties input_props;
  std::vector<hbDNNTensorProperties> output_props;

  int input_h = 0;
  int input_w = 0;
  int output_count = 0;
  std::array<int, 3> output_order{{0, 1, 2}};

  std::array<std::pair<float, float>, 3> s_anchors{{
    {10.0f, 13.0f}, {16.0f, 30.0f}, {33.0f, 23.0f}}};
  std::array<std::pair<float, float>, 3> m_anchors{{
    {30.0f, 61.0f}, {62.0f, 45.0f}, {59.0f, 119.0f}}};
  std::array<std::pair<float, float>, 3> l_anchors{{
    {116.0f, 90.0f}, {156.0f, 198.0f}, {373.0f, 326.0f}}};

  Impl(const Params & p) : params(p)
  {
    if (params.model_path.empty()) {
      throw std::invalid_argument("yolo model_path is empty");
    }

    const char * model_file_name = params.model_path.c_str();
    hbCheck(hbDNNInitializeFromFiles(&packed, &model_file_name, 1), "hbDNNInitializeFromFiles");

    const char ** model_name_list = nullptr;
    int model_count = 0;
    hbCheck(hbDNNGetModelNameList(&model_name_list, &model_count, packed), "hbDNNGetModelNameList");
    if (model_count <= 0) {
      throw std::runtime_error("No model in packed bin");
    }

    hbCheck(hbDNNGetModelHandle(&model, packed, model_name_list[0]), "hbDNNGetModelHandle");

    int32_t input_count = 0;
    hbCheck(hbDNNGetInputCount(&input_count, model), "hbDNNGetInputCount");
    if (input_count != 1) {
      throw std::runtime_error("Expect 1 input");
    }

    hbCheck(hbDNNGetInputTensorProperties(&input_props, model, 0), "hbDNNGetInputTensorProperties");
    if (input_props.tensorType != HB_DNN_IMG_TYPE_NV12) {
      throw std::runtime_error("Expect NV12 input");
    }
    if (input_props.tensorLayout != HB_DNN_LAYOUT_NCHW) {
      throw std::runtime_error("Expect NCHW input");
    }
    if (input_props.validShape.numDimensions != 4) {
      throw std::runtime_error("Expect 4D input (NCHW)");
    }

    input_h = static_cast<int>(input_props.validShape.dimensionSize[2]);
    input_w = static_cast<int>(input_props.validShape.dimensionSize[3]);

    input.properties = input_props;
    const int input_bytes = (input_h * input_w * 3) / 2;
    hbCheck(
      hbSysAllocCachedMem(&input.sysMem[0], std::max(input_bytes, input_props.alignedByteSize)),
      "hbSysAllocCachedMem(input)");

    int32_t out_count = 0;
    hbCheck(hbDNNGetOutputCount(&out_count, model), "hbDNNGetOutputCount");
    output_count = out_count;
    if (output_count <= 0) {
      throw std::runtime_error("No outputs");
    }

    outputs.resize(output_count);
    output_props.resize(output_count);
    for (int i = 0; i < output_count; ++i) {
      hbDNNTensorProperties & out_props = outputs[i].properties;
      hbCheck(hbDNNGetOutputTensorProperties(&out_props, model, i), "hbDNNGetOutputTensorProperties");
      output_props[i] = out_props;
      hbSysMem & mem = outputs[i].sysMem[0];
      hbCheck(hbSysAllocCachedMem(&mem, out_props.alignedByteSize), "hbSysAllocCachedMem(output)");
    }

    if (output_count != 3) {
      throw std::runtime_error("Expect 3 YOLO output heads");
    }

    if (params.anchors.size() == 18) {
      for (int i = 0; i < 3; ++i) {
        s_anchors[static_cast<size_t>(i)] = {
          params.anchors[static_cast<size_t>(i * 2)],
          params.anchors[static_cast<size_t>(i * 2 + 1)]};
        m_anchors[static_cast<size_t>(i)] = {
          params.anchors[static_cast<size_t>(6 + i * 2)],
          params.anchors[static_cast<size_t>(6 + i * 2 + 1)]};
        l_anchors[static_cast<size_t>(i)] = {
          params.anchors[static_cast<size_t>(12 + i * 2)],
          params.anchors[static_cast<size_t>(12 + i * 2 + 1)]};
      }
    }

    const int h32 = input_h / 32;
    const int w32 = input_w / 32;
    const int h16 = input_h / 16;
    const int w16 = input_w / 16;
    const int h8 = input_h / 8;
    const int w8 = input_w / 8;
    const int c = 3 * kFeatureDim;

    int expected_shapes[3][3] = {
      {h32, w32, c},
      {h16, w16, c},
      {h8, w8, c}};

    for (int i = 0; i < 3; ++i) {
      bool found = false;
      for (int j = 0; j < 3; ++j) {
        const auto & shape = outputs[j].properties.validShape;
        if (shape.numDimensions != 4) {
          continue;
        }
        const int d1 = shape.dimensionSize[1];
        const int d2 = shape.dimensionSize[2];
        const int d3 = shape.dimensionSize[3];
        const bool nhwc_ok = (d1 == expected_shapes[i][0] && d2 == expected_shapes[i][1] && d3 == expected_shapes[i][2]);
        const bool nchw_ok = (d1 == expected_shapes[i][2] && d2 == expected_shapes[i][0] && d3 == expected_shapes[i][1]);
        if (nhwc_ok || nchw_ok) {
          output_order[i] = j;
          found = true;
          break;
        }
      }
      if (!found) {
        throw std::runtime_error("Cannot map YOLO outputs to 32/16/8 heads");
      }
    }
  }

  ~Impl()
  {
    for (auto & t : outputs) {
      hbSysFreeMem(&t.sysMem[0]);
    }
    hbSysFreeMem(&input.sysMem[0]);

    if (packed) {
      hbDNNRelease(packed);
      packed = nullptr;
    }
  }

  static std::array<cv::Point2f, 4> reorderKptsToTLBLBRTR(const std::array<cv::Point2f, 4> & pts)
  {
    std::array<cv::Point2f, 4> sorted = pts;
    std::sort(sorted.begin(), sorted.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
      if (std::abs(a.y - b.y) < 1e-4f) {
        return a.x < b.x;
      }
      return a.y < b.y;
    });

    std::array<cv::Point2f, 2> top = {sorted[0], sorted[1]};
    std::array<cv::Point2f, 2> bottom = {sorted[2], sorted[3]};
    if (top[0].x > top[1].x) {
      std::swap(top[0], top[1]);
    }
    if (bottom[0].x > bottom[1].x) {
      std::swap(bottom[0], bottom[1]);
    }

    return {top[0], bottom[0], bottom[1], top[1]};
  }

  static void letterboxImage(
    const cv::Mat & img, int dst_w, int dst_h, cv::Mat & out,
    float & scale, int & x_shift, int & y_shift)
  {
    if (img.empty() || img.type() != CV_8UC3) {
      throw std::invalid_argument("infer expects CV_8UC3 image");
    }

    if (img.cols == dst_w && img.rows == dst_h) {
      scale = 1.0f;
      x_shift = 0;
      y_shift = 0;
      out = img;
      return;
    }

    scale = std::min(1.0f * dst_h / img.rows, 1.0f * dst_w / img.cols);
    if (scale <= 0.0f) {
      throw std::runtime_error("Invalid scale");
    }

    int new_w = static_cast<int>(std::round(img.cols * scale));
    int new_h = static_cast<int>(std::round(img.rows * scale));

    x_shift = (dst_w - new_w) / 2;
    y_shift = (dst_h - new_h) / 2;

    int x_other = dst_w - new_w - x_shift;
    int y_other = dst_h - new_h - y_shift;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    cv::copyMakeBorder(
      resized, out, y_shift, y_other, x_shift, x_other,
      cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
  }

  void fillInputNV12(const cv::Mat & bgr_letterboxed)
  {
    cv::Mat yuv_i420;
    cv::cvtColor(bgr_letterboxed, yuv_i420, cv::COLOR_BGR2YUV_I420);

    uint8_t * dst_nv12 = reinterpret_cast<uint8_t *>(input.sysMem[0].virAddr);
    const uint8_t * src_i420 = yuv_i420.ptr<uint8_t>();

    const int y_size = input_h * input_w;
    const int uv_h = input_h / 2;
    const int uv_w = input_w / 2;
    std::memcpy(dst_nv12, src_i420, static_cast<size_t>(y_size));

    uint8_t * dst_uv = dst_nv12 + y_size;
    const uint8_t * src_u = src_i420 + y_size;
    const uint8_t * src_v = src_u + uv_h * uv_w;
    for (int i = 0; i < uv_h * uv_w; ++i) {
      *dst_uv++ = *src_u++;
      *dst_uv++ = *src_v++;
    }
  }

  const float * outputAsFloat(const hbDNNTensor & out_tensor, std::vector<float> & deq) const
  {
    const auto qtype = out_tensor.properties.quantiType;
    const auto & vshape = out_tensor.properties.validShape;
    int64_t total_elems = 1;
    for (int i = 0; i < vshape.numDimensions; ++i) {
      total_elems *= vshape.dimensionSize[i];
    }

    if (qtype == SHIFT) {
      deq.resize(static_cast<size_t>(total_elems));
      const int8_t * src = reinterpret_cast<int8_t *>(out_tensor.sysMem[0].virAddr);
      const int shift = (out_tensor.properties.shift.shiftData ? out_tensor.properties.shift.shiftData[0] : 0);
      const float s = 1.0f / static_cast<float>(1 << shift);
      for (int64_t i = 0; i < total_elems; ++i) {
        deq[static_cast<size_t>(i)] = src[i] * s;
      }
      return deq.data();
    }
    if (qtype == SCALE) {
      deq.resize(static_cast<size_t>(total_elems));
      const int8_t * src = reinterpret_cast<int8_t *>(out_tensor.sysMem[0].virAddr);
      const float s =
        (out_tensor.properties.scale.scaleData ? out_tensor.properties.scale.scaleData[0] : 1.0f);
      for (int64_t i = 0; i < total_elems; ++i) {
        deq[static_cast<size_t>(i)] = src[i] * s;
      }
      return deq.data();
    }
    return reinterpret_cast<float *>(out_tensor.sysMem[0].virAddr);
  }

  template<typename AnchorArray>
  void processFeatureMap(
    const hbDNNTensor & out_tensor,
    int height, int width,
    const AnchorArray & anchors,
    const cv::Mat & orig_rgb,
    float scale, int x_shift, int y_shift,
    std::vector<cv::Rect> & nms_boxes,
    std::vector<float> & nms_scores,
    std::vector<Yolo::Detection> & raw_dets)
  {
    hbSysFlushMem(const_cast<hbSysMem *>(&out_tensor.sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    const auto & shape = out_tensor.properties.validShape;
    if (shape.numDimensions != 4) {
      throw std::runtime_error("Unexpected output dims");
    }

    const int d1 = shape.dimensionSize[1];
    const int d2 = shape.dimensionSize[2];
    const int d3 = shape.dimensionSize[3];
    const int channels = static_cast<int>(anchors.size()) * kFeatureDim;

    const bool is_nhwc = (d1 == height && d2 == width && d3 == channels);
    const bool is_nchw = (d1 == channels && d2 == height && d3 == width);
    if (!is_nhwc && !is_nchw) {
      throw std::runtime_error("Output shape mismatch for feature map");
    }

    std::vector<float> deq;
    const float * raw_data = outputAsFloat(out_tensor, deq);

    auto get_value = [&](int anchor_idx, int feat_idx, int h_idx, int w_idx) -> float {
      const int c_idx = anchor_idx * kFeatureDim + feat_idx;
      if (is_nhwc) {
        const int index = ((h_idx * width + w_idx) * channels) + c_idx;
        return raw_data[index];
      }
      const int index = ((c_idx * height + h_idx) * width) + w_idx;
      return raw_data[index];
    };

    const float stride = static_cast<float>(input_h) / static_cast<float>(height);
    const float inv_scale = (scale > 0.0f ? 1.0f / scale : 1.0f);

    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (size_t anchor_idx = 0; anchor_idx < anchors.size(); ++anchor_idx) {
          const float obj = sigmoid(get_value(static_cast<int>(anchor_idx), 8, h, w));
          if (obj < params.score_threshold) {
            continue;
          }

          int color_idx = 0;
          for (int i = 1; i < kColorClassCount; ++i) {
            if (get_value(static_cast<int>(anchor_idx), 9 + i, h, w) >
              get_value(static_cast<int>(anchor_idx), 9 + color_idx, h, w))
            {
              color_idx = i;
            }
          }

          if (color_idx >= 2) {
            continue;
          }
          if (params.detect_color == 0 && color_idx == 1) {
            continue;
          }
          if (params.detect_color == 1 && color_idx == 0) {
            continue;
          }

          int cls_idx = 0;
          float cls_max = get_value(static_cast<int>(anchor_idx), 13, h, w);
          for (int i = 1; i < kClassClassCount; ++i) {
            const float v = get_value(static_cast<int>(anchor_idx), 13 + i, h, w);
            if (v > cls_max) {
              cls_max = v;
              cls_idx = i;
            }
          }

          std::array<cv::Point2f, 4> points;
          for (int pt = 0; pt < 4; ++pt) {
            const float decoded_x =
              get_value(static_cast<int>(anchor_idx), pt * 2, h, w) * anchors[anchor_idx].first +
              static_cast<float>(w) * stride;
            const float decoded_y =
              get_value(static_cast<int>(anchor_idx), pt * 2 + 1, h, w) * anchors[anchor_idx].second +
              static_cast<float>(h) * stride;

            float ox = (decoded_x - static_cast<float>(x_shift)) * inv_scale;
            float oy = (decoded_y - static_cast<float>(y_shift)) * inv_scale;
            ox = std::min(std::max(ox, 0.0f), static_cast<float>(orig_rgb.cols - 1));
            oy = std::min(std::max(oy, 0.0f), static_cast<float>(orig_rgb.rows - 1));
            points[pt] = cv::Point2f(ox, oy);
          }

          std::array<cv::Point2f, 4> kpts_raw = {points[0], points[3], points[2], points[1]};
          std::array<cv::Point2f, 4> kpts = reorderKptsToTLBLBRTR(kpts_raw);

          float minx = kpts[0].x;
          float maxx = kpts[0].x;
          float miny = kpts[0].y;
          float maxy = kpts[0].y;
          for (int i = 1; i < 4; ++i) {
            minx = std::min(minx, kpts[i].x);
            maxx = std::max(maxx, kpts[i].x);
            miny = std::min(miny, kpts[i].y);
            maxy = std::max(maxy, kpts[i].y);
          }

          cv::Rect rect(
            static_cast<int>(minx), static_cast<int>(miny),
            static_cast<int>(std::max(0.0f, maxx - minx)),
            static_cast<int>(std::max(0.0f, maxy - miny)));
          if (rect.width <= 0 || rect.height <= 0) {
            continue;
          }

          Yolo::Detection det;
          det.kpts = kpts;
          det.color_idx = color_idx;
          det.cls_idx = cls_idx;
          det.obj = obj;
          det.cls_score = sigmoid(cls_max);
          det.score = det.obj * det.cls_score;

          if (det.score < params.score_threshold) {
            continue;
          }

          raw_dets.emplace_back(det);
          nms_boxes.emplace_back(rect);
          nms_scores.emplace_back(det.score);
        }
      }
    }
  }

  std::vector<Yolo::Detection> postprocess(
    const cv::Mat & orig_rgb,
    float scale, int x_shift, int y_shift)
  {
    std::vector<Yolo::Detection> raw_dets;
    std::vector<cv::Rect> nms_boxes;
    std::vector<float> nms_scores;

    processFeatureMap(
      outputs.at(output_order[0]), input_h / 32, input_w / 32, l_anchors,
      orig_rgb, scale, x_shift, y_shift, nms_boxes, nms_scores, raw_dets);
    processFeatureMap(
      outputs.at(output_order[1]), input_h / 16, input_w / 16, m_anchors,
      orig_rgb, scale, x_shift, y_shift, nms_boxes, nms_scores, raw_dets);
    processFeatureMap(
      outputs.at(output_order[2]), input_h / 8, input_w / 8, s_anchors,
      orig_rgb, scale, x_shift, y_shift, nms_boxes, nms_scores, raw_dets);

    std::vector<int> indices;
    cv::dnn::NMSBoxes(
      nms_boxes, nms_scores, params.score_threshold, params.nms_threshold,
      indices, 1.0f, params.nms_top_k);

    std::vector<Yolo::Detection> kept;
    kept.reserve(indices.size());
    for (int idx : indices) {
      if (idx >= 0 && idx < static_cast<int>(raw_dets.size())) {
        kept.push_back(raw_dets[idx]);
      }
    }

    return kept;
  }
};

Yolo::Yolo(const Params & params) : impl_(std::make_unique<Impl>(params))
{
}

Yolo::~Yolo() = default;

std::vector<Yolo::Detection> Yolo::infer(const cv::Mat & bgr)
{
  using clock = std::chrono::steady_clock;
  auto t0 = clock::now();

  float scale = 1.0f;
  int x_shift = 0;
  int y_shift = 0;

  cv::Mat letterboxed;
  Impl::letterboxImage(bgr, impl_->input_w, impl_->input_h, letterboxed, scale, x_shift, y_shift);

  impl_->fillInputNV12(letterboxed);

  auto t1 = clock::now();

  hbSysFlushMem(&impl_->input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

  hbDNNInferCtrlParam infer_ctrl_param;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

  hbDNNTaskHandle_t task_handle = nullptr;

  hbDNNTensor * out_ptrs = impl_->outputs.data();
  hbCheck(
    hbDNNInfer(&task_handle, &out_ptrs, &impl_->input, impl_->model, &infer_ctrl_param),
    "hbDNNInfer");
  hbCheck(hbDNNWaitTaskDone(task_handle, 0), "hbDNNWaitTaskDone");

  auto t2 = clock::now();

  auto dets = impl_->postprocess(bgr, scale, x_shift, y_shift);

  auto t3 = clock::now();

  {
    std::lock_guard<std::mutex> lock(impl_->timings_mutex);
    impl_->last_timings.preprocess_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
    impl_->last_timings.infer_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    impl_->last_timings.postprocess_ms =
      std::chrono::duration<double, std::milli>(t3 - t2).count();
    impl_->last_timings.src_w = bgr.cols;
    impl_->last_timings.src_h = bgr.rows;
    impl_->last_timings.input_w = impl_->input_w;
    impl_->last_timings.input_h = impl_->input_h;
    impl_->last_timings.scale = scale;
    impl_->last_timings.x_shift = x_shift;
    impl_->last_timings.y_shift = y_shift;
    impl_->last_timings.letterbox_used = !(bgr.cols == impl_->input_w && bgr.rows == impl_->input_h);
  }

  hbDNNReleaseTask(task_handle);
  return dets;
}

void Yolo::setDetectColor(int detect_color)
{
  impl_->params.detect_color = detect_color;
}

void Yolo::setScoreThreshold(float score_threshold)
{
  impl_->params.score_threshold = score_threshold;
}

void Yolo::setNmsThreshold(float nms_threshold)
{
  impl_->params.nms_threshold = nms_threshold;
}

void Yolo::setNmsTopK(int nms_top_k)
{
  impl_->params.nms_top_k = nms_top_k;
}

void Yolo::setAnchors(const std::vector<float> & anchors)
{
  impl_->params.anchors = anchors;
  if (anchors.size() != 18) {
    return;
  }
  for (int i = 0; i < 3; ++i) {
    impl_->s_anchors[static_cast<size_t>(i)] = {
      anchors[static_cast<size_t>(i * 2)],
      anchors[static_cast<size_t>(i * 2 + 1)]};
    impl_->m_anchors[static_cast<size_t>(i)] = {
      anchors[static_cast<size_t>(6 + i * 2)],
      anchors[static_cast<size_t>(6 + i * 2 + 1)]};
    impl_->l_anchors[static_cast<size_t>(i)] = {
      anchors[static_cast<size_t>(12 + i * 2)],
      anchors[static_cast<size_t>(12 + i * 2 + 1)]};
  }
}

const Yolo::Timings & Yolo::lastTimings() const
{
  return impl_->last_timings;
}

}  // namespace rm_auto_aim
