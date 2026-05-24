#include "../include/vision_serial_driver/vision_serial_driver_node.hpp"

serial_driver_node::serial_driver_node(std::string device_name, std::string node_name)
    : rclcpp::Node(node_name), vArray{new visionArray}, rArray{new robotArray},
      dev_name{new std::string(device_name)},
      portConfig{new SerialPortConfig(115200, FlowControl::NONE, Parity::NONE, StopBits::ONE)}, ctx{IoContext(2)}
{
  RCLCPP_INFO(get_logger(), "节点:/%s启动", node_name.c_str());
  muzzleSpeedFilter.Size=10;
  // 清零
  memset(vArray->array, 0, sizeof(visionArray));
  memset(rArray->array, 0, sizeof(robotArray));
  // 设置重启计时器1hz.
  reopenTimer = create_wall_timer(
      1s, std::bind(&serial_driver_node::serial_reopen_callback, this));
  // 设置发布计时器500hz.
  publishTimer = create_wall_timer(
      2ms, std::bind(&serial_driver_node::robot_callback, this));

  // TF broadcaster
  timestamp_offset_ = this->declare_parameter("timestamp_offset", 0.006);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // Detect parameter client
  detector_param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, "armor_detector");

  // 发布Robot信息.
  publisher = create_publisher<vision_interfaces::msg::Robot>(
      "/serial_driver/robot", rclcpp::SensorDataQoS());

  // 订阅AutoAim信息.
  autoAimSub = create_subscription<vision_interfaces::msg::AutoAim>(
      "/serial_driver/aim_target", rclcpp::SensorDataQoS(), std::bind(&serial_driver_node::auto_aim_callback, this, std::placeholders::_1));

  // 设置串口读取线程.
  serialReadThread = std::thread(&serial_driver_node::serial_read_thread, this);
  serialReadThread.detach();
}

serial_driver_node::~serial_driver_node()
{
  if (serialDriver.port()->is_open())
  {
    serialDriver.port()->close();
  }
}

void serial_driver_node::serial_reopen_callback()
{
  // 串口失效时尝试重启0
  if (!isOpen)
  {
    try
    {
      RCLCPP_WARN(get_logger(), "重启串口:%s...", dev_name->c_str());
      serialDriver.init_port(*dev_name, *portConfig);
      serialDriver.port()->open();
      isOpen = serialDriver.port()->is_open();
    }
    catch (const std::system_error &error)
    {
      RCLCPP_ERROR(get_logger(), "打开串口:%s失败", dev_name->c_str());
      isOpen = false;
    }
    if (isOpen)
      RCLCPP_INFO(get_logger(), "打开串口:%s成功", dev_name->c_str());
  }
}

void serial_driver_node::serial_read_thread()
{
  while (rclcpp::ok())
  {
    std::vector<uint8_t> head(2);
    std::vector<uint8_t> robotData(sizeof(rArray->array) - 2);
    if (isOpen)
    {
      try
      {
        serialDriver.port()->receive(head);
        if (head[0] == 0xA5 && head[1] == 0x00)
        { // 包头为0xA5
          serialDriver.port()->receive(robotData);
          robotData.resize(sizeof(rArray->array));
          robotData.insert(robotData.begin(), head[1]);
          robotData.insert(robotData.begin(), head[0]);
          float lastSpeed = rArray->msg.muzzleSpeed;
          memcpy(rArray->array, robotData.data(), sizeof(rArray->array));
          if(rArray->msg.muzzleSpeed != lastSpeed)muzzleSpeedFilter.update(rArray->msg.muzzleSpeed);
          // RCLCPP_INFO(get_logger(), "读取串口.");
        }
      }
      catch (const std::exception &error)
      {
        RCLCPP_ERROR(get_logger(), "读取串口时发生错误.");
        isOpen = false;
      }
    }
  }
}

void serial_driver_node::serial_write(uint8_t *data, size_t len)
{
  std::vector<uint8_t> tempData(data, data + len);
  try
  {
    serialDriver.port()->send(tempData);
    // RCLCPP_INFO(get_logger(), "写入串口.");
  }
  catch (const std::exception &error)
  {
    RCLCPP_ERROR(get_logger(), "写入串口时发生错误.");
    isOpen = false;
  }
}

void serial_driver_node::auto_aim_callback(const vision_interfaces::msg::AutoAim vMsg)
{
  if (isOpen)
  {
    vArray->msg.head = 0xA5;
    vArray->msg.fire = vMsg.fire;
    vArray->msg.aimPitch = vMsg.aim_pitch;
    vArray->msg.aimYaw = vMsg.aim_yaw;
    vArray->msg.tracking = vMsg.tracking;
    serial_write(vArray->array, sizeof(vArray->array));
  }
}

void serial_driver_node::robot_callback()
{
  if (isOpen)
  {
    try
    {
      auto msg = vision_interfaces::msg::Robot();
//msg.foe_color=rArray->msg.foeColor;
      //msg.foe_color=1; //1打红色，我方蓝，0打蓝色我方红
      msg.foe_color=0;
      msg.mode = rArray->msg.mode;
      msg.self_yaw = rArray->msg.robotYaw;
      msg.self_pitch = rArray->msg.robotPitch;
      double muzzle_speed = rArray->msg.muzzleSpeed;

      muzzleSpeedFilter.get_avg(muzzle_speed);
      if (muzzle_speed < 1)
        {
          muzzle_speed = 11.0;
        }
      msg.muzzle_speed = muzzle_speed;



      publisher->publish(msg);

      geometry_msgs::msg::TransformStamped t;
      timestamp_offset_ = this->get_parameter("timestamp_offset").as_double();
      t.header.stamp = this->now() + rclcpp::Duration::from_seconds(timestamp_offset_);
      t.header.frame_id = "odom";
      t.child_frame_id = "gimbal_link";
      tf2::Quaternion q;
      q.setRPY(0, -rArray->msg.robotPitch*3.1415926535/180.0, rArray->msg.robotYaw*3.1415926535/180.0);
      t.transform.rotation = tf2::toMsg(q);
      tf_broadcaster_->sendTransform(t);

      if (!initial_set_param_ || msg.foe_color != previous_receive_color_) {
          
        auto param = rclcpp::Parameter("detect_color", msg.foe_color);

        if (!detector_param_client_->wait_for_service(std::chrono::seconds(5))) {
          RCLCPP_WARN(get_logger(), "Service not ready, skipping parameter set");
          return;
        }

        if (
          !set_param_future_.valid() ||
          set_param_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
          RCLCPP_INFO(get_logger(), "Setting detect_color to %ld...", param.as_int());
          set_param_future_ = detector_param_client_->set_parameters(
            {param}, [this, param](const ResultFuturePtr & results) {
              for (const auto & result : results.get()) {
                if (!result.successful) {
                  RCLCPP_ERROR(get_logger(), "Failed to set parameter: %s", result.reason.c_str());
                  return;
                }
              }
              RCLCPP_INFO(get_logger(), "Successfully set detect_color to %ld!", param.as_int());
              initial_set_param_ = true;
            });
        }
        previous_receive_color_ = msg.foe_color;
      }
    }
    catch (const std::exception &ex)
    {
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 20, "处理串口数据时发生错误: %s", ex.what());
    }
  }
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  std::string dev_name = "/dev/ttyACM0";
  std::shared_ptr<serial_driver_node> node = std::make_shared<serial_driver_node>(dev_name, "vision_serial_driver");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}