#ifndef MULTITAK_TYPES_H
#define MULTITAK_TYPES_H

#include <vector>
#include <map>

namespace hiai {
// 关键点
struct FaceLandmark {
    uint32_t pointNum{0};
    std::vector<float> x{0};
    std::vector<float> y{0};
    std::vector<float> conf{0};
};

// 角度
struct FaceAngle {
    float pitch{0.0f};
    float yaw{0.0f};
    float roll{0.0f};
};

// 遮挡
struct FaceOcclusion {
    float leftEye{0.0f};
    float rightEye{0.0f};
    float nose{0.0f};
    float mouth{0.0f};
};

// 性别
struct FaceGender {
    float male{0.0f};
    float female{0.0f};
};

// 肤色
struct FaceRace {
    float white{0.0f};
    float black{0.0f};
    float asian{0.0f};
};

// 表情
struct FaceExpression {
    float calm{0.0f};
    float cry{0.0f};
    float smile{0.0f};
    float surprise{0.0f};
};

// 睁闭眼
struct FaceEyeStatus {
    float open{0.0f};
    float close{0.0f};
};

// 分割
struct FaceSegmentation {
    std::vector<uint8_t> mask{0};
    uint32_t maskLen{0};
    int8_t maskType = -1;
};

struct DetectBox {
    float leftRatio{0.0f};
    float topRatio{0.0f};
    float rightRatio{0.0f};
    float bottomRatio{0.0f};
    float confidence{0.0};
    int type{-1};
    int id{-1};
};

struct SceneSegmentationResult {
    std::vector<uint8_t> mask{0};
    uint32_t width{0};
    uint32_t height{0};
};

struct FaceInfo {
    DetectBox box;
    FaceLandmark landmark;
    FaceAngle angle;
    FaceOcclusion occlusion;
    std::map<float, float> ageMap;  // key为年龄阶段，value为置信度
    FaceGender gender;
    FaceRace race;
    FaceExpression expression;
    FaceEyeStatus eyeStatus;
    FaceSegmentation segmentation;
};

struct MultitaskResult {
    std::vector<FaceInfo> faceInfos;
    std::vector<DetectBox> objectInfos;
    SceneSegmentationResult sceneSegmentationResult;
};
}  // namespace hiai
#endif