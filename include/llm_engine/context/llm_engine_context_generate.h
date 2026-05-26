/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_GENERATE_H
#define LLM_ENGINE_CONTEXT_GENERATE_H

#include "common/lm_engine_buf.h"
#include "context/llm_engine_context_def.h"
#include "kvcache/llm_engine_kvcache_def.h"
#include "llm_engine_return_types.h"


#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetMaxGenTokens(
    HIAI_LLMEngine_Context* ctx,
    uint32_t m
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetMaxGenTokens(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* m
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetStopSeq(
    HIAI_LLMEngine_Context* ctx,
    char** stopSeq,
    uint32_t stopSeqLen
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetGenerateStatus(
    const HIAI_LLMEngine_Context* ctx
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOneGenerationLen(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOneGeneration(
    const HIAI_LLMEngine_Context* ctx,
    char* generation,
    uint32_t len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllGenerationLen(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllGeneration(
    const HIAI_LLMEngine_Context* ctx,
    char* generation,
    uint32_t len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetOneTokenGeneration(
    const HIAI_LLMEngine_Context* ctx,
    int32_t* genToken
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllTokenGenerationLen(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAllTokenGeneration(
    const HIAI_LLMEngine_Context* ctx,
    int32_t* genToken,
    uint32_t genTokenLen
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillOneTokenGeneration(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t len,
    uint32_t* indices,
    float* logits
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillAllTokenGeneration(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t len,
    uint32_t* indices,
    float* logits
);

// audioBuf直接返回给用户，在下一次推理前用户要拷贝走，不然会覆盖
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetAudioGeneration(
    const HIAI_LLMEngine_Context* ctx,
    void** audioBuf,
    uint64_t* audioBufSizeOfBytes
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_TerminateOnce(
    HIAI_LLMEngine_Context* ctx
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetInitTokenLen(
    HIAI_LLMEngine_Context* ctx,
    uint32_t initTokenLen
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetInitTokenLen(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* initTokenLen
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetKVCache(
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_KVCache* kvCache
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_GENERATE_H
