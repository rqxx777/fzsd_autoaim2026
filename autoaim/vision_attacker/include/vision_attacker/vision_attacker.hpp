/**
  ****************************(C) COPYRIGHT 2024 Polarbear*************************
  * @file       vision_attacker.hpp
  * @brief      
  * @note       
  * @history
  *  Version    Date            Author          Modification
  *  V1.0.0     2024-5          prom-se         1. done(原作者是西安理工大学的仓库中的开源)
  *  V1.0.1     2024-9-3        ZikangXie       1. 修改为RV格式的节点构造，和本队伍的所需话题对齐

  *
  @verbatim
  =================================================================================

  =================================================================================
  @endverbatim
  ****************************(C) COPYRIGHT 2024 Polarbear*************************
  */

#include <rclcpp/rclcpp.hpp>
#include <vision_interfaces/msg/auto_aim.hpp>
#include <vision_interfaces/msg/robot.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "../include/vision_attacker/outpost.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

namespace vision_attacker
{
class VisionAttacker : public rclcpp::Node
{
public:
  explicit VisionAttacker(const rclcpp::NodeOptions & options);

  virtual ~VisionAttacker() override;

private:
  void robot_callback(const vision_interfaces::msg::Robot robot);

  void target_callback(const auto_aim_interfaces::msg::Target target_msg);

  double solveTrajectory(
    double & flyTime, const double x, const double z, const double lagTime,
    const double muzzleSpeed, const double air_k);
  double lagTime = 0.08;
  double airK = 0.04;
  double yawFix = 0;
  double pitchFix = 0;
  double thresholdFix = 0;
  double carThreshold = 30.0;
  double outpostThreshold = 5.0;
  visualization_msgs::msg::Marker aimPoint;
  visualization_msgs::msg::Marker realAimPoint;
  rclcpp::Subscription<auto_aim_interfaces::msg::Target>::SharedPtr targetSub;
  std::unique_ptr<vision_interfaces::msg::Robot> robotPtr;
  rclcpp::Subscription<vision_interfaces::msg::Robot>::SharedPtr robotSub;
  rclcpp::Publisher<vision_interfaces::msg::AutoAim>::SharedPtr aimPub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr markerPub;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr aimMarkerPub;
};
}  // namespace vision_attacker
