// Copyright (C) 2026
// Licensed under the MIT License.

#include "armor_detector/yolo.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#ifdef ARMOR_DETECTOR_USE_HOBOT_CV
#include <hobot_cv/hobotcv_imgproc.h>
#include <rcutils/logging.h>
#endif

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

#ifdef ARMOR_DETECTOR_USE_HOBOT_CV
static void configureHobotCvLogLevelOnce()
{
  static bool configured = false;
  if (!configured) {
    configured = true;
    const rcutils_ret_t ret =
      rcutils_logging_set_logger_level("hobot_cv", RCUTILS_LOG_SEVERITY_WARN);
    (void)ret;
  }
}
#endif

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

  // Reusable buffers to reduce per-frame allocations
  cv::Mat letterboxed_buf;
  cv::Mat yuv_i420_buf;
  cv::Mat y_plane_buf;
  cv::Mat uv422_buf;
  cv::Mat y_resized_buf;
  cv::Mat uv422_resized_buf;
  std::vector<float> dequant_scale_buf;
  std::vector<Yolo::Detection> raw_dets_buf;
  std::vector<cv::Rect> nms_boxes_buf;
  std::vector<float> nms_scores_buf;
  std::vector<int> nms_indices_buf;
  std::vector<int> pre_nms_order_buf;
  std::vector<Yolo::Detection> filtered_dets_buf;
  std::vector<cv::Rect> filtered_boxes_buf;
  std::vector<float> filtered_scores_buf;

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
          params.anchors[static_cast<size_t>(i * 2 +  1)]};
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

    constexpr size_t kMaxCandidates = static_cast<size_t>((80 * 80 + 40 * 40 + 20 * 20) * 3);
    raw_dets_buf.reserve(kMaxCandidates);
    nms_boxes_buf.reserve(kMaxCandidates);
    nms_scores_buf.reserve(kMaxCandidates);
    nms_indices_buf.reserve(params.nms_top_k > 0 ? static_cast<size_t>(params.nms_top_k) : kMaxCandidates);
    pre_nms_order_buf.reserve(kMaxCandidates);
    filtered_dets_buf.reserve(kMaxCandidates);
    filtered_boxes_buf.reserve(kMaxCandidates);
    filtered_scores_buf.reserve(kMaxCandidates);
    dequant_scale_buf.reserve(static_cast<size_t>(3 * kFeatureDim));
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

  static void calcLetterboxParams(
    int src_w, int src_h, int dst_w, int dst_h,
    float & scale, int & x_shift, int & y_shift, int & new_w, int & new_h)
  {
    scale = std::min(1.0f * dst_h / src_h, 1.0f * dst_w / src_w);
    if (scale <= 0.0f) {
      throw std::runtime_error("Invalid scale");
    }

    new_w = std::max(2, static_cast<int>(std::round(src_w * scale)));
    new_h = std::max(2, static_cast<int>(std::round(src_h * scale)));

    if (new_w > dst_w) {
      new_w = dst_w;
    }
    if (new_h > dst_h) {
      new_h = dst_h;
    }

    if (new_w % 2 != 0) {
      new_w -= 1;
    }
    if (new_h % 2 != 0) {
      new_h -= 1;
    }

    x_shift = (dst_w - new_w) / 2;
    y_shift = (dst_h - new_h) / 2;

    if (x_shift % 2 != 0) {
      x_shift -= 1;
    }
    if (y_shift % 2 != 0) {
      y_shift -= 1;
    }
  }

  void fillInputNV12(const cv::Mat & bgr_letterboxed)
  {
    cv::cvtColor(bgr_letterboxed, yuv_i420_buf, cv::COLOR_BGR2YUV_I420);

    uint8_t * dst_nv12 = reinterpret_cast<uint8_t *>(input.sysMem[0].virAddr);
    const uint8_t * src_i420 = yuv_i420_buf.ptr<uint8_t>();

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

  void fillInputNV12FromBGR(const cv::Mat & bgr, float & scale, int & x_shift, int & y_shift)
  {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
      throw std::invalid_argument("infer expects CV_8UC3 image");
    }

    letterboxImage(bgr, input_w, input_h, letterboxed_buf, scale, x_shift, y_shift);

#ifdef ARMOR_DETECTOR_USE_HOBOT_CV
    configureHobotCvLogLevelOnce();
    if ((letterboxed_buf.cols % 2) == 0 && (letterboxed_buf.rows % 2) == 0) {
      cv::Mat nv12;
      const int color_ret =
        hobot_cv::hobotcv_color(letterboxed_buf, nv12, hobot_cv::DCOLOR_BGR2YUV_NV12);
      if (color_ret == 0 && !nv12.empty()) {
        std::memcpy(input.sysMem[0].virAddr, nv12.data, static_cast<size_t>(input_h * input_w * 3 / 2));
        return;
      }
    }
#endif

    // Fallback path when hobot_cv is unavailable or conversion fails.
    fillInputNV12(letterboxed_buf);
  }

  void fillInputNV12FromYUYV(const cv::Mat & yuyv, float & scale, int & x_shift, int & y_shift)
  {
    if (yuyv.empty() || yuyv.type() != CV_8UC2) {
      throw std::invalid_argument("infer expects CV_8UC2 YUYV image");
    }

    int new_w = 0;
    int new_h = 0;
    calcLetterboxParams(yuyv.cols, yuyv.rows, input_w, input_h, scale, x_shift, y_shift, new_w, new_h);

    y_plane_buf.create(yuyv.rows, yuyv.cols, CV_8UC1);
    uv422_buf.create(yuyv.rows, yuyv.cols / 2, CV_8UC2);

    for (int r = 0; r < yuyv.rows; ++r) {
      const uint8_t * src = yuyv.ptr<uint8_t>(r);
      uint8_t * y_dst = y_plane_buf.ptr<uint8_t>(r);
      cv::Vec2b * uv_dst = uv422_buf.ptr<cv::Vec2b>(r);
      for (int c = 0; c < yuyv.cols / 2; ++c) {
        const int idx = c * 4;
        y_dst[2 * c] = src[idx];
        y_dst[2 * c + 1] = src[idx + 2];
        uv_dst[c][0] = src[idx + 1];
        uv_dst[c][1] = src[idx + 3];
      }
    }

    y_resized_buf.create(new_h, new_w, CV_8UC1);
    uv422_resized_buf.create(new_h, new_w / 2, CV_8UC2);
    cv::resize(y_plane_buf, y_resized_buf, cv::Size(new_w, new_h), 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(uv422_buf, uv422_resized_buf, cv::Size(new_w / 2, new_h), 0.0, 0.0, cv::INTER_LINEAR);

    uint8_t * dst_nv12 = reinterpret_cast<uint8_t *>(input.sysMem[0].virAddr);
    const int y_size = input_h * input_w;
    uint8_t * dst_y = dst_nv12;
    uint8_t * dst_uv = dst_nv12 + y_size;

    std::memset(dst_y, 127, static_cast<size_t>(y_size));
    std::memset(dst_uv, 128, static_cast<size_t>(y_size / 2));

    for (int r = 0; r < new_h; ++r) {
      uint8_t * dst_row = dst_y + (r + y_shift) * input_w + x_shift;
      const uint8_t * src_row = y_resized_buf.ptr<uint8_t>(r);
      std::memcpy(dst_row, src_row, static_cast<size_t>(new_w));
    }

    for (int r = 0; r < new_h / 2; ++r) {
      uint8_t * dst_row = dst_uv + (r + y_shift / 2) * input_w + x_shift;
      const cv::Vec2b * uv_row0 = uv422_resized_buf.ptr<cv::Vec2b>(2 * r);
      const cv::Vec2b * uv_row1 = uv422_resized_buf.ptr<cv::Vec2b>(std::min(2 * r + 1, new_h - 1));
      for (int c = 0; c < new_w / 2; ++c) {
        const uint8_t u = static_cast<uint8_t>((static_cast<int>(uv_row0[c][0]) + static_cast<int>(uv_row1[c][0]) + 1) / 2);
        const uint8_t v = static_cast<uint8_t>((static_cast<int>(uv_row0[c][1]) + static_cast<int>(uv_row1[c][1]) + 1) / 2);
        dst_row[2 * c] = u;
        dst_row[2 * c + 1] = v;
      }
    }
  }

  template<typename AnchorArray>
  void processFeatureMap(
    const hbDNNTensor & out_tensor,
    int height, int width,
    const AnchorArray & anchors,
    int orig_w, int orig_h,
    float scale, int x_shift, int y_shift,
    std::vector<cv::Rect> & nms_boxes,
    std::vector<float> & nms_scores,
    std::vector<Yolo::Detection> & raw_dets)
  {
    hbSysFlushMem(const_cast<hbSysMem *>(&out_tensor.sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

    const auto & vshape = out_tensor.properties.validShape;
    const auto & ashape = out_tensor.properties.alignedShape;
    if (vshape.numDimensions != 4 || ashape.numDimensions != 4) {
      throw std::runtime_error("Unexpected output dims");
    }

    const int d1 = vshape.dimensionSize[1];
    const int d2 = vshape.dimensionSize[2];
    const int d3 = vshape.dimensionSize[3];
    const int channels = static_cast<int>(anchors.size()) * kFeatureDim;

    const bool is_nhwc = (d1 == height && d2 == width && d3 == channels);
    const bool is_nchw = (d1 == channels && d2 == height && d3 == width);
    if (!is_nhwc && !is_nchw) {
      throw std::runtime_error("Output shape mismatch for feature map");
    }

    const int aligned_h = ashape.dimensionSize[1];
    const int aligned_w = ashape.dimensionSize[2];
    const int aligned_c = ashape.dimensionSize[3];
    if (is_nhwc && (aligned_h < height || aligned_w < width || aligned_c < channels)) {
      throw std::runtime_error("Aligned NHWC shape smaller than valid area");
    }
    if (is_nchw && (ashape.dimensionSize[1] < channels || ashape.dimensionSize[2] < height || ashape.dimensionSize[3] < width)) {
      throw std::runtime_error("Aligned NCHW shape smaller than valid area");
    }

    const auto & props = out_tensor.properties;
    const int tensor_type = props.tensorType;
    const bool use_scale = (props.scale.scaleData != nullptr);
    dequant_scale_buf.assign(static_cast<size_t>(channels), 1.0f);
    if (props.quantiType != NONE) {
      if (use_scale) {
        const int scale_len = props.scale.scaleLen > 0 ? props.scale.scaleLen : 1;
        for (int c = 0; c < channels; ++c) {
          const int scale_idx = (scale_len > 1) ? (c % scale_len) : 0;
          dequant_scale_buf[static_cast<size_t>(c)] = props.scale.scaleData[scale_idx];
        }
      } else {
        const int shift_len = props.shift.shiftLen > 0 ? props.shift.shiftLen : 1;
        for (int c = 0; c < channels; ++c) {
          const int shift_idx = (shift_len > 1) ? (c % shift_len) : 0;
          dequant_scale_buf[static_cast<size_t>(c)] = std::ldexp(1.0f, -props.shift.shiftData[shift_idx]);
        }
      }
    }

    auto * raw_int32 = reinterpret_cast<int32_t *>(out_tensor.sysMem[0].virAddr);
    auto * raw_int16 = reinterpret_cast<int16_t *>(out_tensor.sysMem[0].virAddr);
    auto * raw_int8 = reinterpret_cast<int8_t *>(out_tensor.sysMem[0].virAddr);
    auto * raw_float = reinterpret_cast<float *>(out_tensor.sysMem[0].virAddr);

    const int aligned_nchw_h = ashape.dimensionSize[2];
    const int aligned_nchw_w = ashape.dimensionSize[3];
    const float stride = static_cast<float>(input_h) / static_cast<float>(height);
    const float inv_scale = (scale > 0.0f ? 1.0f / scale : 1.0f);
    const float score_threshold_safe = std::min(std::max(params.score_threshold, 1e-6f), 1.0f - 1e-6f);
    const float obj_logit_thres = -std::log((1.0f / score_threshold_safe) - 1.0f);

#define RUN_DECODE_LOOP(READ_EXPR, INDEX_EXPR) \
    do { \
      auto read_value = [&](int c_idx, int h_idx, int w_idx) -> float { \
        const int index = (INDEX_EXPR); \
        return (READ_EXPR); \
      }; \
      for (int h = 0; h < height; ++h) { \
        for (int w = 0; w < width; ++w) { \
          for (size_t anchor_idx = 0; anchor_idx < anchors.size(); ++anchor_idx) { \
            const int base_c = static_cast<int>(anchor_idx) * kFeatureDim; \
            const float obj_logit = read_value(base_c + 8, h, w); \
            if (obj_logit < obj_logit_thres) { \
              continue; \
            } \
            const float obj = sigmoid(obj_logit); \
            if (obj < params.score_threshold) { \
              continue; \
            } \
            int color_idx = 0; \
            float best_color_logit = read_value(base_c + 9, h, w); \
            for (int i = 1; i < kColorClassCount; ++i) { \
              const float color_logit = read_value(base_c + 9 + i, h, w); \
              if (color_logit > best_color_logit) { \
                best_color_logit = color_logit; \
                color_idx = i; \
              } \
            } \
            if (color_idx >= 2) { \
              continue; \
            } \
            if (params.detect_color == 0 && color_idx == 1) { \
              continue; \
            } \
            if (params.detect_color == 1 && color_idx == 0) { \
              continue; \
            } \
            int cls_idx = 0; \
            float cls_max = read_value(base_c + 13, h, w); \
            for (int i = 1; i < kClassClassCount; ++i) { \
              const float v = read_value(base_c + 13 + i, h, w); \
              if (v > cls_max) { \
                cls_max = v; \
                cls_idx = i; \
              } \
            } \
            std::array<cv::Point2f, 4> points; \
            for (int pt = 0; pt < 4; ++pt) { \
              const float decoded_x = read_value(base_c + pt * 2, h, w) * anchors[anchor_idx].first + static_cast<float>(w) * stride; \
              const float decoded_y = read_value(base_c + pt * 2 + 1, h, w) * anchors[anchor_idx].second + static_cast<float>(h) * stride; \
              float ox = (decoded_x - static_cast<float>(x_shift)) * inv_scale; \
              float oy = (decoded_y - static_cast<float>(y_shift)) * inv_scale; \
              ox = std::min(std::max(ox, 0.0f), static_cast<float>(orig_w - 1)); \
              oy = std::min(std::max(oy, 0.0f), static_cast<float>(orig_h - 1)); \
              points[pt] = cv::Point2f(ox, oy); \
            } \
            std::array<cv::Point2f, 4> kpts_raw = {points[0], points[3], points[2], points[1]}; \
            std::array<cv::Point2f, 4> kpts = reorderKptsToTLBLBRTR(kpts_raw); \
            float minx = kpts[0].x; \
            float maxx = kpts[0].x; \
            float miny = kpts[0].y; \
            float maxy = kpts[0].y; \
            for (int i = 1; i < 4; ++i) { \
              minx = std::min(minx, kpts[i].x); \
              maxx = std::max(maxx, kpts[i].x); \
              miny = std::min(miny, kpts[i].y); \
              maxy = std::max(maxy, kpts[i].y); \
            } \
            cv::Rect rect( \
              static_cast<int>(minx), static_cast<int>(miny), \
              static_cast<int>(std::max(0.0f, maxx - minx)), \
              static_cast<int>(std::max(0.0f, maxy - miny))); \
            if (rect.width <= 0 || rect.height <= 0) { \
              continue; \
            } \
            Yolo::Detection det; \
            det.kpts = kpts; \
            det.color_idx = color_idx; \
            det.cls_idx = cls_idx; \
            det.obj = obj; \
            det.cls_score = sigmoid(cls_max); \
            det.score = det.obj * det.cls_score; \
            if (det.score < params.score_threshold) { \
              continue; \
            } \
            raw_dets.emplace_back(det); \
            nms_boxes.emplace_back(rect); \
            nms_scores.emplace_back(det.score); \
          } \
        } \
      } \
    } while (0)

    if (is_nhwc) {
      switch (tensor_type) {
        case HB_DNN_TENSOR_TYPE_S32:
          RUN_DECODE_LOOP(static_cast<float>(raw_int32[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((h_idx * aligned_w + w_idx) * aligned_c + c_idx));
          break;
        case HB_DNN_TENSOR_TYPE_S16:
          RUN_DECODE_LOOP(static_cast<float>(raw_int16[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((h_idx * aligned_w + w_idx) * aligned_c + c_idx));
          break;
        case HB_DNN_TENSOR_TYPE_S8:
          RUN_DECODE_LOOP(static_cast<float>(raw_int8[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((h_idx * aligned_w + w_idx) * aligned_c + c_idx));
          break;
        case HB_DNN_TENSOR_TYPE_F32:
          RUN_DECODE_LOOP(raw_float[index],
                          ((h_idx * aligned_w + w_idx) * aligned_c + c_idx));
          break;
        default:
          RUN_DECODE_LOOP(static_cast<float>(raw_int32[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((h_idx * aligned_w + w_idx) * aligned_c + c_idx));
          break;
      }
    } else {
      switch (tensor_type) {
        case HB_DNN_TENSOR_TYPE_S32:
          RUN_DECODE_LOOP(static_cast<float>(raw_int32[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((c_idx * aligned_nchw_h + h_idx) * aligned_nchw_w + w_idx));
          break;
        case HB_DNN_TENSOR_TYPE_S16:
          RUN_DECODE_LOOP(static_cast<float>(raw_int16[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((c_idx * aligned_nchw_h + h_idx) * aligned_nchw_w + w_idx));
          break;
        case HB_DNN_TENSOR_TYPE_S8:
          RUN_DECODE_LOOP(static_cast<float>(raw_int8[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((c_idx * aligned_nchw_h + h_idx) * aligned_nchw_w + w_idx));
          break;
        case HB_DNN_TENSOR_TYPE_F32:
          RUN_DECODE_LOOP(raw_float[index],
                          ((c_idx * aligned_nchw_h + h_idx) * aligned_nchw_w + w_idx));
          break;
        default:
            RUN_DECODE_LOOP(static_cast<float>(raw_int32[index]) * dequant_scale_buf[static_cast<size_t>(c_idx)],
                          ((c_idx * aligned_nchw_h + h_idx) * aligned_nchw_w + w_idx));
          break;
      }
    }

#undef RUN_DECODE_LOOP
  }

  std::vector<Yolo::Detection> postprocess(
    int orig_w, int orig_h,
    float scale, int x_shift, int y_shift)
  {
    raw_dets_buf.clear();
    nms_boxes_buf.clear();
    nms_scores_buf.clear();
    nms_indices_buf.clear();

    processFeatureMap(
      outputs.at(output_order[0]), input_h / 32, input_w / 32, l_anchors,
      orig_w, orig_h, scale, x_shift, y_shift, nms_boxes_buf, nms_scores_buf, raw_dets_buf);
    processFeatureMap(
      outputs.at(output_order[1]), input_h / 16, input_w / 16, m_anchors,
      orig_w, orig_h, scale, x_shift, y_shift, nms_boxes_buf, nms_scores_buf, raw_dets_buf);
    processFeatureMap(
      outputs.at(output_order[2]), input_h / 8, input_w / 8, s_anchors,
      orig_w, orig_h, scale, x_shift, y_shift, nms_boxes_buf, nms_scores_buf, raw_dets_buf);

    if (params.pre_nms_top_k > 0 && static_cast<int>(raw_dets_buf.size()) > params.pre_nms_top_k) {
      pre_nms_order_buf.resize(raw_dets_buf.size());
      std::iota(pre_nms_order_buf.begin(), pre_nms_order_buf.end(), 0);
      auto conf_greater = [&](int a, int b) {
        return nms_scores_buf[static_cast<size_t>(a)] > nms_scores_buf[static_cast<size_t>(b)];
      };
      std::nth_element(
        pre_nms_order_buf.begin(), pre_nms_order_buf.begin() + params.pre_nms_top_k,
        pre_nms_order_buf.end(), conf_greater);
      pre_nms_order_buf.resize(static_cast<size_t>(params.pre_nms_top_k));

      filtered_dets_buf.clear();
      filtered_boxes_buf.clear();
      filtered_scores_buf.clear();
      filtered_dets_buf.reserve(pre_nms_order_buf.size());
      filtered_boxes_buf.reserve(pre_nms_order_buf.size());
      filtered_scores_buf.reserve(pre_nms_order_buf.size());
      for (int idx : pre_nms_order_buf) {
        filtered_dets_buf.push_back(raw_dets_buf[static_cast<size_t>(idx)]);
        filtered_boxes_buf.push_back(nms_boxes_buf[static_cast<size_t>(idx)]);
        filtered_scores_buf.push_back(nms_scores_buf[static_cast<size_t>(idx)]);
      }
      raw_dets_buf.swap(filtered_dets_buf);
      nms_boxes_buf.swap(filtered_boxes_buf);
      nms_scores_buf.swap(filtered_scores_buf);
    }

    cv::dnn::NMSBoxes(
      nms_boxes_buf, nms_scores_buf, params.score_threshold, params.nms_threshold,
      nms_indices_buf, 1.0f, params.nms_top_k);

    std::vector<Yolo::Detection> kept;
    kept.reserve(nms_indices_buf.size());
    for (int idx : nms_indices_buf) {
      if (idx >= 0 && idx < static_cast<int>(raw_dets_buf.size())) {
        kept.push_back(raw_dets_buf[idx]);
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
  const int src_w = bgr.cols;
  const int src_h = bgr.rows;

  if (bgr.type() == CV_8UC3) {
    impl_->fillInputNV12FromBGR(bgr, scale, x_shift, y_shift);
  } else if (bgr.type() == CV_8UC2) {
    impl_->fillInputNV12FromYUYV(bgr, scale, x_shift, y_shift);
  } else {
    throw std::invalid_argument("infer expects CV_8UC3(BGR) or CV_8UC2(YUYV)");
  }

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

  auto dets = impl_->postprocess(src_w, src_h, scale, x_shift, y_shift);

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

void Yolo::setPreNmsTopK(int pre_nms_top_k)
{
  impl_->params.pre_nms_top_k = pre_nms_top_k;
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
