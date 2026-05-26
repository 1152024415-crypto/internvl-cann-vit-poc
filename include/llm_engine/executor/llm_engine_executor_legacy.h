/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_EXECUTOR_LEGACY_H
#define LLM_ENGINE_EXECUTOR_LEGACY_H

#include "common/lm_engine_model_info.h"
#include "executor/llm_engine_executor_def.h"
#include "kvcache/llm_engine_kvcache_legacy.h"
#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SavePromptKVCache(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_PromptKVCache* promptKVCache
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_LoadPromptKVCache(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_PromptKVCache* promptKVCache
);

typedef struct HIAI_LLMEngine_ModelInfo {
    HIAI_LLMEngine_TokenizerType tokenizerType;
    const char* tokenizerPath;
    HIAI_LLMEngine_ModelType modelType;
    const char* modelPath;
} HIAI_LLMEngine_ModelInfo;
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Init(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_ModelInfo* modelInfo
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitSpecModel(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_InitOption* initOption
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitSpecModel(
    HIAI_LLMEngine_Executor* executor
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitVisionModel(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LMEngine_ModelInfo* modelInfo
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitVisionModel(
    HIAI_LLMEngine_Executor* executor
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitVisionProjection(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LMEngine_ModelInfo* modelInfo
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitVisionProjection(
    HIAI_LLMEngine_Executor* executor
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitCompressModel(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_InitOption* initOption
);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitCompressModel(
    HIAI_LLMEngine_Executor* executor
);

typedef enum {
    /** 数据类型为uint8_t */
    HIAI_LLMENGINE_DATATYPE_UINT8 = 0,
} HIAI_LLMEngine_DataType;
 
typedef enum {
    /** RGBA8888_U8格式的图片 */
    HIAI_LLMENGINE_RGBA8888_U8 = 0,
    /** RGB888_U8格式的图片 */
    HIAI_LLMENGINE_RGB888_U8 = 1,
    /** BGR888_U8格式的图片 */
    HIAI_LLMENGINE_BGR888_U8 = 2,
    HIAI_LLMENGINE_IMAGE_FORMAT_INVALID = 255,
} HIAI_LLMEngine_ImageFormat;
 
typedef struct {
    int64_t channel;
    int64_t height;
    int64_t width;
    HIAI_LLMEngine_DataType dataType;
    HIAI_LLMEngine_ImageFormat imageFormat;
} HIAI_LLMEngine_Image_Info;
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetImageInfos(
    HIAI_LLMEngine_Prompt* prompt,
    HIAI_LLMEngine_Image_Info imgInfos[],
    uint32_t imgNum
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_LEAGCY_H
