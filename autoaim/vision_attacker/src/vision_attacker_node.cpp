#include <rclcpp/rclcpp.hpp>
#include <vision_interfaces/msg/robot.hpp>
#include <vision_interfaces/msg/auto_aim.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include "auto_aim_interfaces/msg/target.hpp"
#include "../include/vision_attacker/outpost.hpp"
#include "Eigen/Dense"

class Tracker_node : public rclcpp::Node
{
public:
    explicit Tracker_node(std::string node_name) : rclcpp::Node(node_name)
    {
        lagTime = declare_parameter("trajectory.lag_time", 0.08);
        airK = declare_parameter("trajectory.air_k", 0.04);
        yawFix = declare_parameter("trajectory.yaw_fix", 1.0);
        pitchFix = declare_parameter("trajectory.pitch_fix", 0.0);
        carThreshold = declare_parameter("fire_ctrl.car_attack_threshold", 30.0);
        outpostThreshold = declare_parameter("fire_ctrl.outpost_attack_threshold", 5.0);
        thresholdFix = declare_parameter("fire_ctrl.threshold_fix", 0.0);
        robotPtr = std::make_unique<vision_interfaces::msg::Robot>();
        markerPub = this->create_publisher<visualization_msgs::msg::Marker>("/aiming_point", 10);
        aimMarkerPub = this->create_publisher<visualization_msgs::msg::Marker>("/real_aiming_point", 10);
        aimPub = create_publisher<vision_interfaces::msg::AutoAim>(
            "/serial_driver/aim_target", rclcpp::SensorDataQoS());
        robotSub = create_subscription<vision_interfaces::msg::Robot>(
            "/serial_driver/robot", rclcpp::SensorDataQoS(), std::bind(&Tracker_node::robot_callback, this, std::placeholders::_1));
        targetSub = this->create_subscription<auto_aim_interfaces::msg::Target>(
            "/tracker/target", rclcpp::SensorDataQoS(), std::bind(&Tracker_node::target_callback, this, std::placeholders::_1));

        aimPoint.header.frame_id = "odom";
        aimPoint.ns = "aiming_point";
        aimPoint.type = visualization_msgs::msg::Marker::SPHERE;
        aimPoint.action = visualization_msgs::msg::Marker::ADD;
        aimPoint.scale.x = aimPoint.scale.y = aimPoint.scale.z = 0.12;
        aimPoint.color.r = 1.0;
        aimPoint.color.g = 1.0;
        aimPoint.color.b = 1.0;
        aimPoint.color.a = 1.0;
        aimPoint.lifetime = rclcpp::Duration::from_seconds(0.1);

        realAimPoint.header.frame_id = "odom";
        realAimPoint.ns = "real_aiming_point";
        realAimPoint.type = visualization_msgs::msg::Marker::SPHERE;
        realAimPoint.action = visualization_msgs::msg::Marker::ADD;
        realAimPoint.scale.x = realAimPoint.scale.y = realAimPoint.scale.z = 0.12;
        realAimPoint.color.r = 1.0;
        realAimPoint.color.g = 0.0;
        realAimPoint.color.b = 0.0;
        realAimPoint.color.a = 1.0;
        realAimPoint.lifetime = rclcpp::Duration::from_seconds(0.1);
    };

private:
    void robot_callback(const vision_interfaces::msg::Robot robot)
    {
        try
        {
            *robotPtr = robot;
        }
        catch (std::exception &ex)
        {
            RCLCPP_ERROR(get_logger(), "获取机器人信息时发生错误.");
        }
    };

    void target_callback(const auto_aim_interfaces::msg::Target target_msg)
    {
        //count time
        const auto callback_start = std::chrono::steady_clock::now();
        const double recv_latency_ms = (this->now() - target_msg.header.stamp).seconds() * 1000.0;


        lagTime = get_parameter("trajectory.lag_time").as_double();
        airK = get_parameter("trajectory.air_k").as_double();
        yawFix = get_parameter("trajectory.yaw_fix").as_double();
        pitchFix = get_parameter("trajectory.pitch_fix").as_double();
        carThreshold = get_parameter("fire_ctrl.car_attack_threshold").as_double();
        outpostThreshold = get_parameter("fire_ctrl.outpost_attack_threshold").as_double();
        thresholdFix = get_parameter("fire_ctrl.threshold_fix").as_double();
        Eigen::Vector2d xy;
        xy << target_msg.position.x, target_msg.position.y;
        double yDis = target_msg.position.z;
        double xDis = xy.norm();
        double speed = robotPtr->muzzle_speed > 15 ? robotPtr->muzzle_speed : 15.00;
        double flyTime = 0;
        solveTrajectory(flyTime, xDis, yDis, lagTime, speed, 0.04);

        vision_interfaces::msg::AutoAim aim;
        aim.aim_yaw = robotPtr->self_yaw;
        aim.aim_pitch = robotPtr->self_pitch;
        aim.fire = 0;
        if (!target_msg.tracking)
        {
            aim.tracking = 0;
            aimPub->publish(aim);
            // RCLCPP_INFO(get_logger(), "aimYaw:%05.2f/aimPitch:%05.2f", aim.aim_yaw, aim.aim_pitch);
            const auto callback_end = std::chrono::steady_clock::now();
            const double recv_to_aim_pub_ms =
                std::chrono::duration<double, std::milli>(callback_end - callback_start).count();
            // RCLCPP_INFO_THROTTLE(
            //     this->get_logger(), *this->get_clock(), 1000,
            //     "VisionAttacker timing [ms]: recv_latency=%.3f, recv_to_aim_pub=%.3f, total=%.3f",
            //     recv_latency_ms, recv_to_aim_pub_ms, recv_to_aim_pub_ms);
            return;
        }
        else
        {
            double threshold = target_msg.id == "outpost" ? outpostThreshold : carThreshold;
            std::shared_ptr<carStates> states = target_msg.id == "outpost" ? std::make_shared<outpostStates>() : std::make_shared<carStates>();
            states->update(target_msg, robotPtr->self_yaw / 180.0 * M_PI);
            std::vector<armorStates> armors = states->getPreArmor(flyTime);
            std::sort(armors.begin(), armors.end(), [](const auto &armor1, const auto &armor2)
                      { return abs(armor1.delta_yaw) < abs(armor2.delta_yaw); });
            if ((-threshold + thresholdFix) / 180.0 * M_PI < armors[0].delta_yaw && armors[0].delta_yaw < (threshold + thresholdFix) / 180.0 * M_PI)
            {
                aim.aim_yaw = atan2(armors[0].y, armors[0].x) * 180.0 / M_PI;
                aim.aim_yaw = aim.aim_yaw < -180  ? aim.aim_yaw + 360.0
                              : aim.aim_yaw > 180 ? aim.aim_yaw - 360.0
                                                  : aim.aim_yaw;
                xDis = std::sqrt(armors[0].x * armors[0].x + armors[0].y * armors[0].y);
                aim.aim_pitch = solveTrajectory(flyTime, xDis, armors[0].z, lagTime, speed, airK) * 180.0 / M_PI;
                aim.fire = abs(aim.aim_yaw - robotPtr->self_yaw) < 5.0 && abs(aim.aim_pitch - robotPtr->self_pitch) < 5.0 ? 1 : 0;
                aimPoint.pose.position.x = armors[0].x;
                aimPoint.pose.position.y = armors[0].y;
                aimPoint.pose.position.z = armors[0].z;
                realAimPoint.pose.position.x = target_msg.position.x - target_msg.radius_1 * cos(robotPtr->self_yaw / 180.0 * M_PI);
                realAimPoint.pose.position.y = target_msg.position.y - target_msg.radius_1 * sin(robotPtr->self_yaw / 180.0 * M_PI);
                realAimPoint.pose.position.z = target_msg.position.z;
            }
            else
            {
                aim.fire = 0;
                double x, y, z;
                x = target_msg.position.x - target_msg.radius_1 * cos(robotPtr->self_yaw / 180.0 * M_PI);
                y = target_msg.position.y - target_msg.radius_1 * sin(robotPtr->self_yaw / 180.0 * M_PI);
                z = target_msg.position.z;
                xDis = std::sqrt(x * x + y * y);
                aim.aim_yaw = atan2(y, x) * 180.0 / M_PI;
                aim.aim_pitch = solveTrajectory(flyTime, xDis, z, lagTime, speed, airK) * 180.0 / M_PI;
            }
            aimPoint.header.stamp = now();
            markerPub->publish(aimPoint);
            aimMarkerPub->publish(realAimPoint);
            aim.aim_yaw += yawFix;
            aim.aim_pitch += pitchFix;
            aim.tracking = 1;
            aimPub->publish(aim);

            const auto aim_pub_end = std::chrono::steady_clock::now();
            const double recv_to_aim_pub_ms =
                std::chrono::duration<double, std::milli>(aim_pub_end - callback_start).count();
            const double callback_total_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - callback_start)
                    .count();
            // RCLCPP_INFO_THROTTLE(
            //     this->get_logger(), *this->get_clock(), 1000,
            //     "VisionAttacker timing [ms]: recv_latency=%.3f, recv_to_aim_pub=%.3f, total=%.3f",
            //     recv_latency_ms, recv_to_aim_pub_ms, callback_total_ms);
            //RCLCPP_INFO(get_logger(), "aimYaw:%05.2f/aimPitch:%05.2f selfYaw:%05.2f/selfPitch:%05.2f", aim.aim_yaw, aim.aim_pitch,robotPtr->self_yaw,robotPtr->self_pitch);
        }
    };

    double solveTrajectory(double &flyTime, const double x, const double z, const double lagTime, const double muzzleSpeed, const double air_k)
    {
        const double gravity = 9.78;
        double theta;
        double time;
        double aimZ;
        double realZ;
        double targetZ;

        aimZ = z;
        for (size_t i = 0; i < 20; i++)
        {
            theta = atan2(aimZ, x);
            time = (exp(air_k * x) - 1) / (air_k * muzzleSpeed * cos(theta));
            realZ = muzzleSpeed * sin(theta) * time - gravity * (time * time) / 2;
            aimZ = aimZ + (z - realZ);
            if (abs(realZ - z) < 0.001)
            {
                time += lagTime;
                flyTime = time;
                targetZ = aimZ;
                break;
            }
            else
            {
                continue;
            }
        }
        return atan2(targetZ, x);
    }
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

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Tracker_node>("vision_attacker");
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}