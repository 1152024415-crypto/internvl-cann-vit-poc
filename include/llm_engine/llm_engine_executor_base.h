/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine executor
 */
#ifndef LLM_ENGINE_EXECUTOR_BASE_H
#define LLM_ENGINE_EXECUTOR_BASE_H

#include "llm_engine_c_api_export.h"
#include "llm_engine_return_types.h"
#include "llm_engine_context_base.h"
#include "common/lm_engine_model_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_Executor HIAI_LLMEngine_Executor;

HIAI_LLMEngine_Executor* HIAI_LLMEngine_Executor_CreateFromJson(const char* jsonStr);

typedef struct HIAI_LLMEngine_Prompt HIAI_LLMEngine_Prompt;

HIAI_LLMEngine_Prompt* HIAI_LLMEngine_Prompt_Create(void);

void HIAI_LLMEngine_Prompt_Destroy(HIAI_LLMEngine_Prompt** prompt);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_LLM_Generate(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_Prompt* prompt);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_LLM_GenerateAsync(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_Prompt* prompt);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetText(
    HIAI_LLMEngine_Prompt* prompt, const char* text);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetTokenIds(
    HIAI_LLMEngine_Prompt* prompt, uint32_t* inputTokens, uint32_t tokenNum);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Deinit(HIAI_LLMEngine_Executor* executor);

void HIAI_LLMEngine_Executor_Destroy(HIAI_LLMEngine_Executor** executor);


#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_H
