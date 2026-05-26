/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_EXECUTOR_INIT_H
#define LLM_ENGINE_EXECUTOR_INIT_H

#include "common/lm_engine_model_info.h"
#include "executor/llm_engine_executor_def.h"
#include "executor/llm_engine_executor_init_opt.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Init_Use_Option(
    HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_InitOption* initOption
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Deinit(
    HIAI_LLMEngine_Executor* executor);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_H
