/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_GENERATE_ASYNC_H
#define LLM_ENGINE_CONTEXT_GENERATE_ASYNC_H

#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif


HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetCallbackFreq(
    HIAI_LLMEngine_Context* ctx,
    uint32_t callbackFreq
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCallbackFreq(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* callbackFreq
);

typedef void(*callbackFuncType)(const HIAI_LLMEngine_Context*);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx,
    callbackFuncType func
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx,
    callbackFuncType func
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnPrefillGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx,
    callbackFuncType func
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnPrefillSomeTokenGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx,
    callbackFuncType func
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefillCallbackTokenNum(
    HIAI_LLMEngine_Context* ctx,
    uint32_t tokenNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed(
    HIAI_LLMEngine_Context* ctx,
    callbackFuncType func
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_GENERATE_ASYNC_H
