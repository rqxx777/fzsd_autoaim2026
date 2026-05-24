#ifndef SERIAL_PACKET_H
#define SERIAL_PACKET_H

#include <cstdint>
#include <algorithm>

#pragma pack(2)

#define visionMsg inf_visionMsg
#define robotMsg inf_robotMsg

// struct sentry_visionMsg
// {
//     uint8_t head;
//     uint8_t fire;   // 开火标志
//     float aimYaw;   // 目标Yaw
//     float aimPitch; // 目标Pitch
//     float linearX;  // 线速度x
//     float linearY;  // 线速度y
//     float angularZ; // 角速度z
// };

// struct sentry_robotMsg
// {
//     uint8_t head;
//     uint8_t foeColor;  // 敌方颜色 0-blue 1-red
//     float robotYaw;    // 自身Yaw
//     float robotPitch;  // 自身Pitch
//     float muzzleSpeed; // 弹速
// };

struct inf_visionMsg
{
    uint16_t head;
    uint8_t fire;     // 开火标志
    uint8_t tracking; // 跟踪标志
    float aimYaw;     // 目标Yaw
    float aimPitch;   // 目标Pitch

};

struct inf_robotMsg
{
    uint16_t head;
    uint8_t mode;
    uint8_t foeColor;  // 敌方颜色 0-blue 1-red
    float robotYaw;    // 自身Yaw
    float robotPitch;  // 自身Pitch
    float muzzleSpeed; // 弹速
};

// struct hero_visionMsg
// {
//     uint16_t head;
//     uint8_t fire;     // 开火标志
//     uint8_t tracking; // 跟踪标志
//     float aimYaw;     // 目标Yaw
//     float aimPitch;   // 目标Pitch

    
// };

// struct hero_robotMsg
// {
//     uint16_t head;
//     uint8_t foeColor;  // 敌方颜色 0-blue 1-red
//     float robotYaw;    // 自身Yaw
//     float robotPitch;  // 自身Pitch
//     float muzzleSpeed; // 弹速
// };

union visionArray
{
    struct visionMsg msg;
    uint8_t array[sizeof(struct visionMsg)];
};

union robotArray
{
    struct robotMsg msg;
    uint8_t array[sizeof(struct robotMsg)];
};

#pragma pack()

#endif // SERIAL_PACKET_H