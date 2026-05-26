/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_DEF_H
#define LLM_ENGINE_CONTEXT_DEF_H

#ifdef __cplusplus
extern "C" {
#endif

#define HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN    10

#define HIAI_LLM_ENGINE_COMPRESS_FORCE_TOKENS_MAX_LEN    100

typedef struct HIAI_LLMEngine_Context HIAI_LLMEngine_Context;

HIAI_LLMEngine_Context* HIAI_LLMEngine_Context_Create(void);

HIAI_LLMEngine_Context* HIAI_LLMEngine_Context_CreateFromContextJson(
    const char* jsonStr
);

void HIAI_LLMEngine_Context_Destroy(
    HIAI_LLMEngine_Context** ctx
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_DEF_H
