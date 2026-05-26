/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_EXECUTOR_DEF_H
#define LLM_ENGINE_EXECUTOR_DEF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_Executor HIAI_LLMEngine_Executor;

HIAI_LLMEngine_Executor* HIAI_LLMEngine_Executor_Create(void);

HIAI_LLMEngine_Executor* HIAI_LLMEngine_Executor_CreateFromJson(
    const char* jsonStr
);

void HIAI_LLMEngine_Executor_Destroy(
    HIAI_LLMEngine_Executor** executor
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_DEF_H
