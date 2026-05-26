/*
 * @Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * @Description: AIagent基础结构定义
 * @Author: l30053676
 * @Date: 2025-10-28 14:57:34
 */
#ifndef AI_AGENT_DATA_H
#define AI_AGENT_DATA_H
#include <map>
#include <string>
#include <vector>
#include "llm_engine_executor.h"
namespace hiai {

// agent数据类型枚举，用于区分不同的数据类型, 业务类型在context中用event_type区分
enum AgentDataType {
    AGENT_DATA_TYPE_UNKNOWN = 0,
    AGENT_DATA_TYPE_IMAGE,
    AGENT_DATA_TYPE_INFO_MAP,
    AGENT_DATA_TYPE_STRING,
    AGENT_DATA_TYPE_MULTITASK_RESULT,
    AGENT_DATA_TYPE_PS_RESULT,
    AGENT_DATA_TYPE_PS_PARAM,
    AGENT_DATA_TYPE_SMART_CAPTURE_PARAM,
    AGENT_DATA_TYPE_IMG,
    AGENT_DATA_TYPE_NONE,
};

// 内部通用数据传递结构体
struct AgentData {
    void *data = nullptr;
    size_t length = 0;
    AgentDataType dataType = AGENT_DATA_TYPE_UNKNOWN;
};

// 定义返回状态码
enum AIAgentStatus {
    AI_AGENT_SUCCESS = 0,
    AI_AGENT_FAILED = -1,
    AI_AGENT_ERROR_INIT = -2,
    AI_AGENT_INVALID_PARAM = -3,
    AI_AGENT_NOT_SUPPORTED = -4,
    AI_AGENT_NOT_INITIALIZED = -5,
    AI_AGENT_REPEATED_INIT = -6,
};

// 图片结构体
struct Image {
    uint8_t* data = nullptr;
    int height = 0; // 图片的高度，若为YUV420SP_U8则为实际图像高
    int width = 0; // 图片的宽高，若为YUV420SP_U8则为实际图像宽
    int stride = 0;
    int channel = 0;
    HIAI_LLMEngine_ImageFormat format = HIAI_LLMENGINE_RGBA8888_U8;
};

// 通用信息 map
using Info = std::map<std::string, std::string>;
// Agent类型key
constexpr const char *AGENT_CONTEXT_AGENT_TYPE_KEY = "agent_type";
// 事件类型key
constexpr const char *AGENT_CONTEXT_EVENT_TYPE_KEY = "event_type";
// Prompt类型key
constexpr const char *AGENT_CONTEXT_PROMPT_TYPE_KEY = "prompt_type";
constexpr const char *AGENT_CONTEXT_PROMPT_CONTENT_KEY = "prompt_content";
// 维测相关key
constexpr const char *AGENT_CONTEXT_PICTURE_NAME_KEY = "picture_name";
// 模型路径key
constexpr const char *AGENT_MODEL_DIR_KEY = "model_dir";
// 人脸信息key
constexpr const char *AGENT_CONTEXT_FACEBOX_KEY = "face_box";

// Agent类型value
constexpr const char *AGENT_TYPE_IMG_CAPTION = "img_caption";
constexpr const char *AGENT_TYPE_IMG_EDIT_RECOM = "img_edit_recom";
constexpr const char *AGENT_TYPE_MULTITASK = "multitask";
// Agent类型value
constexpr const char *AGENT_TYPE_POSE_SUGGESTION = "PoseSuggestion";
// Agent类型value
constexpr const char *AGENT_TYPE_SMART_CAPTURE = "SmartCapture";
// 事件类型value
constexpr const char *AGENT_EVENT_IMAGE_CAPTION = "image_process";
constexpr const char *AGENT_EVENT_QUERY_VERSION = "query_version";
constexpr const char *AGENT_EVENT_MULTITASK_DETECTION = "multitask_detection";
constexpr const char *AGENT_EVENT_IMAGE_EDIT_RECOM = "image_edit_recom";

// Prompt类型value
constexpr const char *AGENT_PROMPT_TYPE_BRIEF = "brief";
constexpr const char *AGENT_PROMPT_TYPE_DETAILED = "detailed";
constexpr const char *AGENT_PROMPT_TYPE_FORMAT = "format";
constexpr const char *AGENT_PROMPT_TYPE_EMOTION = "emotion";
constexpr const char *AGENT_PROMPT_TYPE_CUSTOMIZED = "customized";
constexpr const char *AGENT_PROMPT_TYPE_RECOMMENDATION = "recommendation";
constexpr const char *AGENT_PROMPT_TYPE_SEMANTICS = "semantics";
constexpr const char *AGENT_PROMPT_TYPE_MULTILIST = "multipromptlist";
// 其他
constexpr const char *AGENT_MODEL_VERSION_UNKNOWN = "Unknown";
}
#endif