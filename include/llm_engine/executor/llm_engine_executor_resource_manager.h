/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_EXECUTOR_RESOURCE_MANAGER_H
#define LLM_ENGINE_EXECUTOR_RESOURCE_MANAGER_H

#include "common/lm_engine_model_info.h"
#include "executor/llm_engine_executor_init_opt_def.h"
#include "executor/llm_engine_executor_def.h"
#include "executor/llm_engine_executor_init_opt.h"
#include "executor/llm_engine_executor_resource_opt_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HIAI_LLMEngine_InferPerf_UNSET = 0,
    HIAI_LLMEngine_InferPerf_LOW,
    HIAI_LLMEngine_InferPerf_MIDDLE,
    HIAI_LLMEngine_InferPerf_HIGH,
    HIAI_LLMEngine_InferPerf_EXTREME_HIGH,
} HIAI_LLMEngine_InferPerfMode;

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SetInferencePerfMode(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_InferPerfMode inferPerfMode
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Suspend(
    HIAI_LLMEngine_Executor* executor
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Resume(
    HIAI_LLMEngine_Executor* executor
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_FreeResource(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_ResourceOption* resourceOption
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_RestoreResource(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_ResourceOption* resourceOption
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetInitOption(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    const HIAI_LLMEngine_InitOption* initOption
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetModelInfo(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    HIAI_LMEngine_ModelInfo* modelInfo
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetSpecModelInfo(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    HIAI_LMEngine_ModelInfo* specModelInfo
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetWeightInfo(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    const char* weightName,
    size_t offset,
    HIAI_LM_Buffer* weightBuffer
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetLoraInfo(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    HIAI_LM_Buffer* configBuffer,
    HIAI_LM_Buffer* loraBuffer,
    HIAI_LM_Buffer* quantBuffer,
    const char* loraConfig
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetSpecWeightInfo(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    const char* weightName,
    size_t offset,
    HIAI_LM_Buffer* weightBuffer
);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetSpecLoraInfo(
    HIAI_LLMEngine_ResourceOption* resourceOption,
    HIAI_LM_Buffer* configBuffer,
    HIAI_LM_Buffer* loraBuffer,
    HIAI_LM_Buffer* quantBuffer,
    const char* loraConfig
);


#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_RESOURCE_MANAGER_H
