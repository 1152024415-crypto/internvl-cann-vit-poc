#ifndef GALLERY_AGENT_INTERFACE_H
#define GALLERY_AGENT_INTERFACE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "ai_agent_data.h"
#include "llm_engine_executor.h"
#include "multitask_types.h"

#ifndef AICP_API_EXPORT
#if !(defined _WIN32) && !(defined __CYGWIN__)
#define AICP_API_EXPORT __attribute__((visibility("default")))
#else
#define AICP_API_EXPORT
#endif
#endif

namespace hiai {
/**
 * @brief 统一的 AI Agent 对外接口类
 * 该接口作为所有Agent业务的统一入口。
 */
class AICP_API_EXPORT IGalleryAgentInterface {
public:
    virtual ~IGalleryAgentInterface() = default;

    /**
     * @brief 初始化Agent
     * @param [in] info 初始化参数, 必须包含 hiai::AGENT_CONTEXT_AGENT_TYPE_KEY 键。
     * @return AIAgentStatus 执行状态
     */
    virtual AIAgentStatus Init(const Info &info) = 0;

    /**
     * @brief 业务接口1: 图像描述处理接口
     * @param [in] image 输入图片
     * @param [in] context 运行时上下文信息。
     *        必须包含 hiai::AGENT_CONTEXT_AGENT_TYPE_KEY 键, 值为 hiai:: AGENT_TYPE_IMG_CAPTION。
     *        必须包含以下用于控制prompt生成的键:
     *        - hiai::AGENT_CONTEXT_PROMPT_TYPE_KEY: 控制 prompt 类型。
     *          - hiai::AGENT_PROMPT_TYPE_BRIEF: 生成一句话的简短描述。
     *          - hiai::AGENT_PROMPT_TYPE_DETAILED: 生成详细描述。
     *          - hiai::AGENT_PROMPT_TYPE_FORMAT: 格式化输出。
     *          - hiai::AGENT_PROMPT_TYPE_EMOTION: 以JSON格式描述图中人物的情绪。
     *          - hiai::AGENT_PROMPT_TYPE_CUSTOMIZED: 使用用户自定义的prompt内容。
     *        - hiai::AGENT_CONTEXT_PROMPT_CONTENT_KEY: string (可选)
     *          - 当 AGENT_CONTEXT_PROMPT_TYPE_KEY 的值为 hiai::AGENT_PROMPT_TYPE_CUSTOMIZED
     * 时，此字段必须存在，其值将作为完整的prompt内容。
     * @param [out] result 处理结果
     * @return AIAgentStatus 执行状态
     */
    virtual AIAgentStatus ImageCaption(const Image &image, const Info &context, Info &result) = 0;

    /**
     * @brief 查询 Agent 模型版本信息接口
     * @param [in] context 附加上下文信息
     *        必须包含 hiai::AGENT_CONTEXT_AGENT_TYPE_KEY 键
     * @param [out] version 用于接收版本信息的字符串
     * @return AIAgentStatus 状态码
     */
    virtual AIAgentStatus QueryVersion(const Info &context, std::string &version) = 0;

    /**
     * @brief 逆初始化Agent
     * @param [in] info 逆初始化参数, 可通过 hiai::AGENT_CONTEXT_AGENT_TYPE_KEY 指定逆初始化某个agent。
     */
    virtual void Deinit(const Info &info) = 0;

    /**
     * @brief 业务接口2: 多任务检测处理接口
     * @param [in] image 输入图片
     * @param [in] context 运行时上下文信息。
     *        必须包含 hiai::AGENT_CONTEXT_AGENT_TYPE_KEY 键, 值为 hiai::AGENT_TYPE_MULTITASK
     *        此外，还可以包含以下键：
     *        - hiai::AGENT_CONTEXT_FACE_INFO_KEY: string，
     *        表示图片中的人脸框坐标，以归一化后浮点数表示，格式为left,top,right,bottom，多个人脸用;分隔。
     * @param [out] result 处理结果
     * @return AIAgentStatus 执行状态
     */
    virtual AIAgentStatus MultitaskDetection(const Image &image, const Info &context, MultitaskResult &result) = 0;

    /**
     * @brief 查询 Agent 模型版本信息接口
     * @param [in] context 运行时上下文信息。
     *        必须包含 hiai::AGENT_CONTEXT_AGENT_TYPE_KEY 键, 值为 hiai:: AGENT_TYPE_IMG_CAPTION。
     *        必须包含以下用于控制prompt生成的键:
     *        - hiai::AGENT_CONTEXT_PROMPT_TYPE_KEY: 控制 prompt 类型。
     *          - hiai::AGENT_PROMPT_TYPE_RECOMMENDATION: 生成修图建议。
     * @param [out] result 处理结果
     * @return AIAgentStatus 状态码
     */
    virtual AIAgentStatus ImgEditRecommendation(const Image &image, const Info &context, Info &result) = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 创建 AI Agent 实例的工厂函数
 * @return 指向 IGalleryAgentInterface 实例的指针, 失败则返回 nullptr
 */
AICP_API_EXPORT hiai::IGalleryAgentInterface *CreateGalleryAgent();

/**
 * @brief 销毁由 CreateAIAgent 创建的实例
 * @param [in] agent 指向 IGalleryAgentInterface 实例的指针
 */
AICP_API_EXPORT void DestroyGalleryAgent(hiai::IGalleryAgentInterface *agent);

#ifdef __cplusplus
}  // extern "C"
#endif

}  // namespace hiai

#endif  // GALLERY_AGENT_INTERFACE_H