/**
  ****************************(C) COPYRIGHT 2024 Polarbear*************************
  * @file       outpost.hpp
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

#ifndef OUT_POST_HPP
#define OUT_POST_HPP

#include <cmath>

#include "auto_aim_interfaces/msg/armors.hpp"
#include "auto_aim_interfaces/msg/target.hpp"

struct armorStates
{
  double x;
  double y;
  double z;
  double delta_yaw;
};

class carStates
{
public:
  virtual void update(const auto_aim_interfaces::msg::Target target_msg, const double self_yaw)
  {
    x = target_msg.position.x;
    y = target_msg.position.y;
    z = target_msg.position.z;
    delta_z = target_msg.dz;
    raduis_1 = target_msg.radius_1;
    raduis_2 = target_msg.radius_2;
    yaw = target_msg.yaw;
    v_x = target_msg.velocity.x;
    v_y = target_msg.velocity.y;
    v_z = target_msg.velocity.z;
    v_yaw = target_msg.v_yaw;
    this->self_yaw = self_yaw;
  }
  virtual std::vector<armorStates> getPreArmor(const double time)
  {
    std::vector<armorStates> preArmors;
    for (size_t i = 0; i < 4; i++) {
      armorStates armor;
      double armor_yaw = yaw + time * v_yaw + M_PI_2 * i;
      while (armor_yaw < 0) {
        armor_yaw += 2 * M_PI;
      };
      while (armor_yaw > 2 * M_PI) {
        armor_yaw -= 2 * M_PI;
      };
      armor.x = x + time * v_x - cos(armor_yaw) * (i % 2 ? raduis_2 : raduis_1);
      armor.y = y + time * v_y - sin(armor_yaw) * (i % 2 ? raduis_2 : raduis_1);
      armor.z = z + time * v_z + (i % 2 ? delta_z : 0.0);
      armor.delta_yaw = armor_yaw - self_yaw;
      armor.delta_yaw = armor.delta_yaw > M_PI ? armor.delta_yaw - M_PI * 2 : armor.delta_yaw;
      preArmors.emplace_back(armor);
    };
    return preArmors;
  }

protected:
  double x;
  double y;
  double z;
  double delta_z;
  double raduis_1;
  double raduis_2;
  double yaw;
  double v_x;
  double v_y;
  double v_z;
  double v_yaw;
  double self_yaw;
};

class outpostStates : public carStates
{
public:
  virtual void update(const auto_aim_interfaces::msg::Target target_msg, const double self_yaw)
  {
    x = target_msg.position.x;
    y = target_msg.position.y;
    z = target_msg.position.z;
    delta_z = 0;
    raduis_1 = 0.553 / 2;
    raduis_2 = raduis_1;
    yaw = target_msg.yaw;
    v_x = 0;
    v_y = 0;
    v_z = 0;
    v_yaw = target_msg.v_yaw > 0.3 ? 0.8 * M_PI : target_msg.v_yaw < -0.3 ? -0.8 * M_PI : 0;
    this->self_yaw = self_yaw;
  }
  virtual std::vector<armorStates> getPreArmor(const double time)
  {
    std::vector<armorStates> preArmors;
    for (size_t i = 0; i < 3; i++) {
      armorStates armor;
      double armor_yaw = yaw + time * v_yaw + 2 * M_PI / 3 * i;
      while (armor_yaw < 0) {
        armor_yaw += 2 * M_PI;
      };
      while (armor_yaw > 2 * M_PI) {
        armor_yaw -= 2 * M_PI;
      };
      armor.x = x + time * v_x - cos(armor_yaw) * raduis_1;
      armor.y = y + time * v_y - sin(armor_yaw) * raduis_1;
      armor.z = z;
      armor.delta_yaw = armor_yaw - self_yaw;
      preArmors.emplace_back(armor);
    };
    return preArmors;
  }
};
#endif  // OUT_POST_HPP