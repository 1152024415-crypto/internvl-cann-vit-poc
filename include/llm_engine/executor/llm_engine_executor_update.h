/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_EXECUTOR_UPDATE_H
#define LLM_ENGINE_EXECUTOR_UPDATE_H

#include "common/lm_engine_model_info.h"
#include "executor/llm_engine_executor_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateWeight(
    HIAI_LLMEngine_Executor* executor,
    const char* weightName,
    size_t offset,
    HIAI_LM_Buffer* weightBuffer
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecWeight(
    HIAI_LLMEngine_Executor* executor,
    const char* weightName,
    size_t offset,
    HIAI_LM_Buffer* weightBuffer
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateLora(
    HIAI_LLMEngine_Executor* executor,
    const char* loraConfig
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateLoraFromBuffer(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LM_Buffer* configBuffer,
    HIAI_LM_Buffer* loraBuffer,
    HIAI_LM_Buffer* quantBuffer
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecLora(
    HIAI_LLMEngine_Executor* executor,
    const char* loraConfig
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecLoraFromBuffer(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LM_Buffer* configBuffer,
    HIAI_LM_Buffer* loraBuffer,
    HIAI_LM_Buffer* quantBuffer
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_UPDATE_H
