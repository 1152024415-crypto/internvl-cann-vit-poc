/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */
#ifndef LLM_ENGINE_EXECUTOR_GENERATE_H
#define LLM_ENGINE_EXECUTOR_GENERATE_H

#include "common/lm_engine_model_info.h"
#include "executor/llm_engine_executor_def.h"
#include "context/llm_engine_context_def.h"
#include "prompt/llm_engine_prompt_def.h"
#include "llm_engine_return_types.h"


#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_CompressPrompt(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    const char* srcPrompt
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Generate(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    const char* prompt
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_GenerateAsync(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    const char* prompt
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MLLMGenerate(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_Prompt* prompt
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MLLMGenerateAsync(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    HIAI_LLMEngine_Prompt* prompt
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_GetSentDataLen(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    uint32_t* len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_GetSentData(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    char* sendData,
    uint32_t len
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SetSparseParam(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    const char* sparseParam
);

typedef enum {
    Prefill = 0,
    Decode
} HIAI_LLMEngine_STAGE;

typedef bool(*setMropeEmbeddingCallback)(
    void* embedding,
    const uint32_t embeddingSize,
    HIAI_LLMEngine_STAGE stage,
    uint32_t genLen
);

typedef bool(*tokenToEmbeddingCallback)(
    int32_t token,
    void* embedding,
    const uint32_t embeddingSize
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MropeEmbeddingGenerate(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    const void* embeddings,
    size_t embeddingsSize,
    const tokenToEmbeddingCallback t2eCallback,
    const setMropeEmbeddingCallback sinEmbeddingCallback,
    const setMropeEmbeddingCallback cosEmbeddingCallback
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MropeEmbeddingGenerateAsync(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx,
    const void* embeddings,
    size_t embeddingsSize,
    const tokenToEmbeddingCallback t2eCallback,
    const setMropeEmbeddingCallback sinEmbeddingCallback,
    const setMropeEmbeddingCallback cosEmbeddingCallback
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_EncodePrompts(
    HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Prompt* prompts[],
    uint32_t promptsSize
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_GENERATE_H
