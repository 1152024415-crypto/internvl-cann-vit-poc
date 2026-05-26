/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_SAMPLE_H
#define LLM_ENGINE_CONTEXT_SAMPLE_H

#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDoSampleFlag(
    HIAI_LLMEngine_Context* ctx,
    bool f
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDoSampleFlag(
    const HIAI_LLMEngine_Context* ctx,
    bool* f
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSeed(
    HIAI_LLMEngine_Context* ctx,
    uint32_t seed
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetSeed(
    const HIAI_LLMEngine_Context* ctx,
    uint32_t* seed
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTopK(
    HIAI_LLMEngine_Context* ctx,
    uint64_t k
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTopK(
    const HIAI_LLMEngine_Context* ctx,
    uint64_t* k
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTopP(
    HIAI_LLMEngine_Context* ctx,
    float p
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTopP(
    const HIAI_LLMEngine_Context* ctx,
    float* p
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTemperature(
    HIAI_LLMEngine_Context* ctx,
    float t
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTemperature(
    const HIAI_LLMEngine_Context* ctx,
    float* t
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetRepetitionPenalty(
    HIAI_LLMEngine_Context* ctx,
    float rp
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetRepetitionPenalty(
    const HIAI_LLMEngine_Context* ctx,
    float* rp
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefillTopk(
    HIAI_LLMEngine_Context* ctx,
    uint32_t topk
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_SAMPLE_H
