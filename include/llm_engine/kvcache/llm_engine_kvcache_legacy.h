/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_KVCACHE_DEF_LEGACY_H
#define LLM_ENGINE_KVCACHE_DEF_LEGACY_H

#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_PromptKVCache HIAI_LLMEngine_PromptKVCache;

HIAI_LLMEngine_PromptKVCache* HIAI_LLMEngine_PromptKVCache_Create(void);

void HIAI_LLMEngine_PromptKVCache_Destory(
    HIAI_LLMEngine_PromptKVCache** promptKVCache
);

HIAI_LLMEngine_PromptKVCache* HIAI_LLMEngine_PromptKVCache_CreateFromBuffer(
    void* data,
    uint64_t size
);

HIAI_LLMEngine_Status HIAI_LLMEngine_PromptKVCache_GetData(
    HIAI_LLMEngine_PromptKVCache* promptKVCache,
    void** data,
    uint64_t* size
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_KVCACHE_DEF_LEGACY_H