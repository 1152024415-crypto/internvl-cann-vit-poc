/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine context extend
 */
#ifndef LLM_ENGINE_CONTEXT_EXTEND_H
#define LLM_ENGINE_CONTEXT_EXTEND_H

#include "llm_engine_c_api_export.h"
#include "llm_engine_return_types.h"
#include "llm_engine_context_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN    10
#define HIAI_LLM_ENGINE_COMPRESS_FORCE_TOKENS_MAX_LEN    100

HIAI_LLMEngine_Context* HIAI_LLMEngine_Context_Create(void);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetCallbackFreq(HIAI_LLMEngine_Context* ctx,
    uint32_t callbackFreq);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCallbackFreq(
    const HIAI_LLMEngine_Context* ctx, uint32_t* callbackFreq);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDoSampleFlag(HIAI_LLMEngine_Context* ctx,
    bool f);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDoSampleFlag(
    const HIAI_LLMEngine_Context* ctx, bool* f);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSeed(HIAI_LLMEngine_Context* ctx,
    uint32_t seed);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetSeed(
    const HIAI_LLMEngine_Context* ctx, uint32_t* seed);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTopK(HIAI_LLMEngine_Context* ctx,
    uint64_t k);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTopK(
    const HIAI_LLMEngine_Context* ctx, uint64_t* k);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTopP(HIAI_LLMEngine_Context* ctx,
    float p);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTopP(
    const HIAI_LLMEngine_Context* ctx, float* p);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTemperature(HIAI_LLMEngine_Context* ctx,
    float t);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTemperature(
    const HIAI_LLMEngine_Context* ctx, float* t);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetRepetitionPenalty(
    HIAI_LLMEngine_Context* ctx, float rp);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetRepetitionPenalty(
    const HIAI_LLMEngine_Context* ctx, float* rp);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetMaxGenTokens(HIAI_LLMEngine_Context* ctx,
    uint32_t m);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetMaxGenTokens(
    const HIAI_LLMEngine_Context* ctx, uint32_t* m);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefixPrompt(
    HIAI_LLMEngine_Context* ctx, const char* prefixPrompt);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_UnSetPrefixPrompt(
    HIAI_LLMEngine_Context* ctx);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetStopSeq(HIAI_LLMEngine_Context* ctx,
    char** stopSeq, uint32_t stopSeqLen);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCachedTokenCount(
    const HIAI_LLMEngine_Context* ctx, uint64_t* count);
    
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnPrefillGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx, callbackFuncType func);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetOnPrefillSomeTokenGenerateDoneFunc(
    HIAI_LLMEngine_Context* ctx, callbackFuncType func);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefillTopk(
    HIAI_LLMEngine_Context* ctx, uint32_t topk);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefillCallbackTokenNum(
    HIAI_LLMEngine_Context* ctx, uint32_t tokenNum);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillOneTokenGeneration(
    const HIAI_LLMEngine_Context* ctx, uint32_t len, uint32_t* indices, float* logits);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillAllTokenGeneration(
    const HIAI_LLMEngine_Context* ctx, uint32_t len, uint32_t* indices, float* logits);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_TerminateOnce(
    HIAI_LLMEngine_Context* ctx);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetInitTokenLen(HIAI_LLMEngine_Context* ctx,
    uint32_t initTokenLen);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetInitTokenLen(
    const HIAI_LLMEngine_Context* ctx, uint32_t* initTokenLen);

// tree spec
typedef enum {
    HIAI_LLMENGINE_STATIC_SPEC_TREE = 0,
    HIAI_LLMENGINE_DYNAMIC_SPEC_TREE = 1,
    HIAI_LLMENGINE_SPEC_TREE_TYPE_INVALID
} HIAI_LLMEngine_SpecTreeType;

typedef enum {
    HIAI_LLMENGINE_GAMMA_DISABLE = 0,
    HIAI_LLMENGINE_GAMMA_EMA_RISE_EMA_FALL = 1,
    HIAI_LLMENGINE_GAMMA_FORCE_RISE_FORCE_FALL = 2,
    HIAI_LLMENGINE_GAMMA_FORCE_RISE_STAIRWAY_FALL = 3,
    HIAI_LLMENGINE_GAMMA_FORCE_RISE_EMA_FALL = 4,
    HIAI_LLMENGINE_GAMMA_STRATEGY_INVALID
} HIAI_LLMEngine_DynGammaStrategy ;

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDynGammaStrategy(HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_DynGammaStrategy strategy);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDynGammaEmaWindow(HIAI_LLMEngine_Context* ctx,
    uint32_t emaWindow);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeConfidenceThresh(HIAI_LLMEngine_Context* ctx,
    uint32_t treeIdx, float thresh);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_ClearSpecTree(HIAI_LLMEngine_Context* ctx);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeType(HIAI_LLMEngine_Context* ctx,
    uint32_t treeIdx, HIAI_LLMEngine_SpecTreeType type);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTree(HIAI_LLMEngine_Context* ctx,
    uint32_t gamma, const uint32_t* topHead, const uint32_t* topKSpec);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeShape(HIAI_LLMEngine_Context* ctx,
    uint32_t treeIdx, uint32_t gamma, const uint32_t* topHead, const uint32_t** topKSpec, uint32_t topKSpecDim);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTypicalScaling(HIAI_LLMEngine_Context* ctx,
    float typicalScaling);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTypicalScaling(HIAI_LLMEngine_Context* ctx,
    float* typicalScaling);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDraftThreshold(HIAI_LLMEngine_Context* ctx,
    float draftThreshold);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDraftThreshold(HIAI_LLMEngine_Context* ctx,
    float* draftThreshold);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSampleGreedy(HIAI_LLMEngine_Context* ctx,
    bool greedy);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetSampleGreedy(HIAI_LLMEngine_Context* ctx,
    bool* greedy);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetRejectScaling(HIAI_LLMEngine_Context* ctx,
    bool rejectScaling);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetRejectScaling(HIAI_LLMEngine_Context* ctx,
    bool* rejectScaling);

// compress
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetCompressRatio(HIAI_LLMEngine_Context* ctx,
    float ratio);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetAdditionalContext(
    HIAI_LLMEngine_Context* ctx, const char* additionalContext);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetForceTokens(HIAI_LLMEngine_Context* ctx,
    char** forceTokens, uint32_t forceTokensLen);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCompressPromptLen(
    const HIAI_LLMEngine_Context* ctx, uint32_t* len);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetCompressPrompt(
    const HIAI_LLMEngine_Context* ctx, char* compressPrompt, uint32_t len);

typedef enum {
    HIAI_LLMENGINE_TREE_SPECTYPE = 0,
    HIAI_LLMENGINE_TEXT_SPECTYPE = 1,
    HIAI_LLMENGINE_UNDEFINED_SPECTYPE
} HIAI_LLMEngine_SpecType;

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecType(HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_SpecType specType);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDraftText(HIAI_LLMEngine_Context* ctx,
    const char* draftText);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetMatchLen(HIAI_LLMEngine_Context* ctx,
    uint32_t matchLen);
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetWindowSize(HIAI_LLMEngine_Context* ctx,
    uint32_t windowSize);

typedef struct HIAI_LLMEngine_KVCache HIAI_LLMEngine_KVCache;
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetKVCache(HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_KVCache* kvCache);
#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_EXTEND_H