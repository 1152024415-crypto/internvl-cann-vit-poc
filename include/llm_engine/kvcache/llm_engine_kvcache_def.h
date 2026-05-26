/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025-2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_KVCACHE_DEF_H
#define LLM_ENGINE_KVCACHE_DEF_H

#include "executor/llm_engine_executor_def.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_KVCache HIAI_LLMEngine_KVCache;

// 内存申请，必须将kvCacheSwitchableFlag设成true
HIAI_LLMEngine_KVCache* HIAI_LLMEngine_KVCache_Create(
    HIAI_LLMEngine_Executor* executor
);

void HIAI_LLMEngine_DestroyKVCache(
    HIAI_LLMEngine_KVCache** kvCache
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_KVCACHE_DEF_H