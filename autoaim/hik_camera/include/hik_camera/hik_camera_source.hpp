#ifndef HIK_CAMERA__HIK_CAMERA_SOURCE_HPP_
#define HIK_CAMERA__HIK_CAMERA_SOURCE_HPP_

#include <chrono>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace hik_camera
{

struct HikFrame
{
  cv::Mat bgr;
  std::chrono::steady_clock::time_point timestamp;
};

class HikCameraSource
{
public:
  HikCameraSource() = default;
  ~HikCameraSource();

  HikCameraSource(const HikCameraSource &) = delete;
  HikCameraSource & operator=(const HikCameraSource &) = delete;

  bool open(double exposure_time, double gain, int timeout_ms, std::string * error = nullptr);
  bool read(HikFrame & frame, std::string * error = nullptr);
  void close();
  bool isOpened() const;

  bool setExposureTime(double exposure_time, std::string * error = nullptr);
  bool setGain(double gain, std::string * error = nullptr);
  void setTimeoutMs(int timeout_ms);

  double exposureTime() const;
  double gain() const;
  int timeoutMs() const;

private:
  void * handle_ = nullptr;
  int timeout_ms_ = 1000;
  double exposure_time_ = 6000.0;
  double gain_ = 10.0;
  std::vector<uint8_t> convert_buffer_;
};

}  // namespace hik_camera

#endif  // HIK_CAMERA__HIK_CAMERA_SOURCE_HPP_
