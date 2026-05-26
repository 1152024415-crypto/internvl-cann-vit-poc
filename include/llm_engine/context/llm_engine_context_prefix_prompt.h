/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_PREFIX_PROMPT_H
#define LLM_ENGINE_CONTEXT_PREFIX_PROMPT_H

#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefixPrompt(
    HIAI_LLMEngine_Context* ctx,
    const char* prefixPrompt
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_UnSetPrefixPrompt(
    HIAI_LLMEngine_Context* ctx
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_PREFIX_PROMPT_H
