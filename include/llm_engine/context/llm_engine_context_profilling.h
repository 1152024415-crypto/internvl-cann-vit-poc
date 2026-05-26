/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_CONTEXT_PROFILLIING_H
#define LLM_ENGINE_CONTEXT_PROFILLIING_H

#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTotalTimeMs(
    const HIAI_LLMEngine_Context* ctx,
    double* ms
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillTimeMs(
    const HIAI_LLMEngine_Context* ctx,
    double* ms
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDecodeTimeMs(
    const HIAI_LLMEngine_Context* ctx,
    double* ms
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetInputTokenCount(
    const HIAI_LLMEngine_Context* ctx,
    uint64_t* count
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOutputTokenCount(
    const HIAI_LLMEngine_Context* ctx,
    uint64_t* count
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCachedTokenCount(
    const HIAI_LLMEngine_Context* ctx,
    uint64_t* count
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDecodeNum(
    const HIAI_LLMEngine_Context* ctx,
    uint64_t* num
);
    
#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_PROFILLIING_H
