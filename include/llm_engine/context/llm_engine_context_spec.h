/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_CONTEXT_SPEC_H
#define LLM_ENGINE_CONTEXT_SPEC_H

#include "context/llm_engine_context_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

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
} HIAI_LLMEngine_DynGammaStrategy;

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDynGammaStrategy(
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_DynGammaStrategy strategy
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDynGammaEmaWindow(
    HIAI_LLMEngine_Context* ctx,
    uint32_t emaWindow
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeConfidenceThresh(
    HIAI_LLMEngine_Context* ctx,
    uint32_t treeIdx,
    float thresh
);
    
HIAI_LLMEngine_Status HIAI_LLMEngine_Context_ClearSpecTree(
    HIAI_LLMEngine_Context* ctx
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeType(
    HIAI_LLMEngine_Context* ctx,
    uint32_t treeIdx,
    HIAI_LLMEngine_SpecTreeType type
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTree(
    HIAI_LLMEngine_Context* ctx,
    uint32_t gamma,
    const uint32_t* topHead,
    const uint32_t* topKSpec
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeShape(
    HIAI_LLMEngine_Context* ctx,
    uint32_t treeIdx,
    uint32_t gamma,
    const uint32_t* topHead,
    const uint32_t** topKSpec,
    uint32_t topKSpecDim
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetTypicalScaling(
    HIAI_LLMEngine_Context* ctx,
    float typicalScaling
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetTypicalScaling(
    HIAI_LLMEngine_Context* ctx,
    float* typicalScaling
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDraftThreshold(
    HIAI_LLMEngine_Context* ctx,
    float draftThreshold
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetDraftThreshold(
    HIAI_LLMEngine_Context* ctx,
    float* draftThreshold
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSampleGreedy(
    HIAI_LLMEngine_Context* ctx,
    bool greedy
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetSampleGreedy(
    HIAI_LLMEngine_Context* ctx,
    bool* greedy
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetRejectScaling(
    HIAI_LLMEngine_Context* ctx,
    bool rejectScaling
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetRejectScaling(
    HIAI_LLMEngine_Context* ctx,
    bool* rejectScaling
);

typedef enum {
    HIAI_LLMENGINE_TREE_SPECTYPE = 0,
    HIAI_LLMENGINE_TEXT_SPECTYPE = 1,
    HIAI_LLMENGINE_UNDEFINED_SPECTYPE
} HIAI_LLMEngine_SpecType;

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecType(
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_SpecType specType
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDraftText(
    HIAI_LLMEngine_Context* ctx,
    const char* draftText
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetMatchLen(
    HIAI_LLMEngine_Context* ctx,
    uint32_t matchLen
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetWindowSize(
    HIAI_LLMEngine_Context* ctx,
    uint32_t windowSize
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_CONTEXT_SPEC_H
