/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_KVCACHE_MANAGER_H
#define LLM_ENGINE_KVCACHE_MANAGER_H

#include "llm_engine_return_types.h"
#include "kvcache/llm_engine_kvcache_def.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_KVCache_GetRealSize(
    HIAI_LLMEngine_KVCache* kvCache,
    uint64_t* size
);

HIAI_LLMEngine_Status HIAI_LLMEngine_KVCache_SaveToBuffer(
    HIAI_LLMEngine_KVCache* kvCache,
    void* data,
    uint64_t size
);

HIAI_LLMEngine_Status HIAI_LLMEngine_KVCache_LoadFromBuffer(
    HIAI_LLMEngine_KVCache* kvCache,
    const void* data,
    uint64_t size
);

HIAI_LLMEngine_Status HIAI_LLMEngine_KVCache_LoadFromFile(
    HIAI_LLMEngine_KVCache* kvCache,
    const char* fileKVCacheName);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_KVCACHE_MANAGER_H