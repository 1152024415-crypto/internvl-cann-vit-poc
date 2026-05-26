/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_PROMPT_TEXT_H
#define LLM_ENGINE_PROMPT_TEXT_H

#include "prompt/llm_engine_prompt_def.h"
#include "llm_engine_return_types.h"
#include "common/lm_engine_buf.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetText(
    HIAI_LLMEngine_Prompt* prompt,
    const char* text
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetTokenIds(
    HIAI_LLMEngine_Prompt* prompt,
    uint32_t* inputTokens,
    uint32_t tokenNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetTextEmbedBufs(
    HIAI_LLMEngine_Prompt* prompt,
    void* textEmbedBufs,
    size_t textEmbedBufsSize,
    uint32_t tokenNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetTextEmbedBufInfo(
    HIAI_LLMEngine_Prompt* prompt,
    HIAI_LMEngine_Tensor_Buf_Info tensorBufInfo
);


#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_PROMPT_TEXT_H
