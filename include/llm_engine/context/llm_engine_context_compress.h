/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_COMPRESS_H
#define LLM_ENGINE_CONTEXT_COMPRESS_H

#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetCompressRatio(
    HIAI_LLMEngine_Context* ctx,
    float ratio
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetAdditionalContext(
    HIAI_LLMEngine_Context* ctx,
    const char* additionalContext
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetForceTokens(
    HIAI_LLMEngine_Context* ctx,
    char** forceTokens,
    uint32_t forceTokensLen
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCompressPromptLen(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCompressPrompt(
    const HIAI_LLMEngine_Context* ctx,
    char* compressPrompt,
    uint32_t len
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_COMPRESS_H
