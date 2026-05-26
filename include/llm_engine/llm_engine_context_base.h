/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine context base
 */
#ifndef LLM_ENGINE_CONTEXT_BASE_H
#define LLM_ENGINE_CONTEXT_BASE_H

#include "llm_engine_c_api_export.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_Context HIAI_LLMEngine_Context;

HIAI_LLMEngine_Context* HIAI_LLMEngine_Context_CreateFromContextJson(const char* jsonStr);

void HIAI_LLMEngine_Context_Destroy(HIAI_LLMEngine_Context** ctx);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTotalTimeMs(
    const HIAI_LLMEngine_Context* ctx, double* ms);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillTimeMs(
    const HIAI_LLMEngine_Context* ctx, double* ms);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDecodeTimeMs(
    const HIAI_LLMEngine_Context* ctx, double* ms);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetInputTokenCount(
    const HIAI_LLMEngine_Context* ctx, uint64_t* count);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOutputTokenCount(
    const HIAI_LLMEngine_Context* ctx, uint64_t* count);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDecodeNum(const HIAI_LLMEngine_Context* ctx,
    uint64_t* num);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetGenerateStatus(
    const HIAI_LLMEngine_Context* ctx);

typedef void(*callbackFuncType)(const HIAI_LLMEngine_Context*);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx, callbackFuncType func);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx, callbackFuncType func);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed(
    HIAI_LLMEngine_Context* ctx, callbackFuncType func);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOneGenerationLen(
    const HIAI_LLMEngine_Context* ctx, uint32_t* len);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOneGeneration(
    const HIAI_LLMEngine_Context* ctx, char* generation, uint32_t len);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllGenerationLen(
    const HIAI_LLMEngine_Context* ctx, uint32_t* len);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllGeneration(
    const HIAI_LLMEngine_Context* ctx, char* generation, uint32_t len);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllTokenGenerationLen(
    const HIAI_LLMEngine_Context* ctx, uint32_t* len);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllTokenGeneration(
    const HIAI_LLMEngine_Context* ctx, int32_t* genToken, uint32_t genTokenLen);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOneTokenGeneration(
    const HIAI_LLMEngine_Context* ctx, int32_t* genToken);
    
#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_BASE_H