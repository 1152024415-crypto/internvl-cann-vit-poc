/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description: llm engine executor
 */
#ifndef LLM_ENGINE_EXECUTOR_EXTEND_H
#define LLM_ENGINE_EXECUTOR_EXTEND_H

#include "llm_engine_c_api_export.h"
#include "llm_engine_return_types.h"
#include "llm_engine_context.h"
#include "llm_engine_executor_base.h"
#include "common/lm_engine_model_info.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_PromptKVCache HIAI_LLMEngine_PromptKVCache;
HIAI_LLMEngine_PromptKVCache* HIAI_LLMEngine_PromptKVCache_Create(void);
void HIAI_LLMEngine_PromptKVCache_Destory(HIAI_LLMEngine_PromptKVCache** promptKVCache);
HIAI_LLMEngine_PromptKVCache* HIAI_LLMEngine_PromptKVCache_CreateFromBuffer(void* data, uint64_t size);
HIAI_LLMEngine_Status HIAI_LLMEngine_PromptKVCache_GetData(
    HIAI_LLMEngine_PromptKVCache* promptKVCache, void** data, uint64_t* size);

typedef struct HIAI_LLMEngine_InitOption HIAI_LLMEngine_InitOption;
HIAI_LLMEngine_InitOption* HIAI_LLMEngine_InitOption_Create(void);
void HIAI_LLMEngine_InitOption_Destroy(HIAI_LLMEngine_InitOption** initOption);

typedef enum {
    HIAI_LLMENGINE_SPM_TOKENIZER = 0,
    HIAI_LLMENGINE_HMB_TOKENIZER = 1,
    HIAI_LLMENGINE_XLMROBERTA_TOKENIZER = 2,
	HIAI_LLMENGINE_BPE_TOKENIZER = 3,
    HIAI_LLMENGINE_QWEN_TOKENIZER = 4,
    HIAI_LLMENGINE_QWEN2_TOKENIZER = 5,
    HIAI_LLMENGINE_GLM_TOKENIZER = 6,
    HIAI_LLMENGINE_UNDEFINED_TOKENIZER = 255,
} HIAI_LLMEngine_TokenizerType;

typedef enum {
    HIAI_LLMENGINE_LLAMA2_MODEL = 0,
    HIAI_LLMENGINE_UNDEFINED_MODEL,
} HIAI_LLMEngine_ModelType;

typedef enum {
    HIAI_LLMENGINE_AUTOREGRESSIVE_INFERTYPE = 0,
    HIAI_LLMENGINE_TREESPEC_INFERTYPE = 1,
    HIAI_LLMENGINE_LLMLINGUA_INFERTYPE = 2,
    HIAI_LLMENGINE_UNDEFINED_INFERTYPE,
} HIAI_LLMEngine_InferType;

typedef struct HIAI_LLMEngine_ModelInfo {
    HIAI_LLMEngine_TokenizerType tokenizerType;
    const char* tokenizerPath;
    HIAI_LLMEngine_ModelType modelType;
    const char* modelPath;
} HIAI_LLMEngine_ModelInfo;

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetInferType(
    HIAI_LLMEngine_InitOption* initOption, HIAI_LLMEngine_InferType inferType);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetTokenizer(
    HIAI_LLMEngine_InitOption* initOption, HIAI_LLMEngine_TokenizerType tokenizerType, const char* tokenizerPath);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetModel(
    HIAI_LLMEngine_InitOption* initOption, HIAI_LLMEngine_ModelType modelType,
    HIAI_LMEngine_ModelInfo* modelInfo);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetSpecModel(
    HIAI_LLMEngine_InitOption* initOption, HIAI_LLMEngine_ModelType specModelType,
    HIAI_LMEngine_ModelInfo* specModelInfo);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Generate(HIAI_LLMEngine_Executor* executor,
    HIAI_LLMEngine_Context* ctx, const char* prompt);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_GenerateAsync(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, const char* prompt);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetKVCacheSwitchableFlag(
    HIAI_LLMEngine_InitOption* initOption, bool kvCacheSwitchableFlag);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetImageBufs(
    HIAI_LLMEngine_Prompt* prompt, void* imgBufs[], size_t imgBufsSize[], uint32_t imgNum);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedImageBufs(
    HIAI_LLMEngine_Prompt* prompt, void* imgEncodedBufs[], size_t imgEncodedBufsSize[], uint32_t imgNum);

HIAI_LLMEngine_Executor* HIAI_LLMEngine_Executor_Create(void);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Init(HIAI_LLMEngine_Executor* executor,
    const HIAI_LLMEngine_ModelInfo* modelInfo);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Init_Use_Option(
    HIAI_LLMEngine_Executor* executor, const HIAI_LLMEngine_InitOption* initOption);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitSpecModel(
    HIAI_LLMEngine_Executor* executor, const HIAI_LLMEngine_InitOption* initOption);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitCompressModel(
    HIAI_LLMEngine_Executor* executor, const HIAI_LLMEngine_InitOption* initOption);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitSpecModel(
    HIAI_LLMEngine_Executor* executor);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitCompressModel(
    HIAI_LLMEngine_Executor* executor);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitVisionModel(
    HIAI_LLMEngine_Executor* executor, const HIAI_LMEngine_ModelInfo* modelInfo);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitVisionModel(
    HIAI_LLMEngine_Executor* executor);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_EncodePrompts(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Prompt* prompts[], uint32_t promptsSize);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitVisionProjection(
    HIAI_LLMEngine_Executor* executor, const HIAI_LMEngine_ModelInfo* modelInfo);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitVisionProjection(
    HIAI_LLMEngine_Executor* executor);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateWeight(
    HIAI_LLMEngine_Executor* executor, const char* weightName, size_t offset, HIAI_LM_Buffer* weightBuffer);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecWeight(
    HIAI_LLMEngine_Executor* executor, const char* weightName, size_t offset, HIAI_LM_Buffer* weightBuffer);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_CompressPrompt(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, const char* srcPrompt);

typedef enum {
    Prefill = 0,
    Decode
} HIAI_LLMEngine_STAGE;

typedef bool(*setMropeEmbeddingCallback)(void* embedding, const uint32_t embeddingSize, HIAI_LLMEngine_STAGE stage,
    uint32_t genLen);
typedef bool(*tokenToEmbeddingCallback)(int32_t token, void* embedding, const uint32_t embeddingSize);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MropeEmbeddingGenerate(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, const void* embeddings, size_t embeddingsSize,
    const tokenToEmbeddingCallback t2eCallback, const setMropeEmbeddingCallback sinEmbeddingCallback,
    const setMropeEmbeddingCallback cosEmbeddingCallback);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MropeEmbeddingGenerateAsync(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, const void* embeddings, size_t embeddingsSize,
    const tokenToEmbeddingCallback t2eCallback, const setMropeEmbeddingCallback sinEmbeddingCallback,
    const setMropeEmbeddingCallback cosEmbeddingCallback);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateLora(HIAI_LLMEngine_Executor* executor,
    const char* loraConfig);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateLoraFromBuffer(
    HIAI_LLMEngine_Executor* executor, HIAI_LM_Buffer* configBuffer, HIAI_LM_Buffer* loraBuffer,
    HIAI_LM_Buffer* quantBuffer);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecLora(HIAI_LLMEngine_Executor* executor,
    const char* loraConfig);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecLoraFromBuffer(
    HIAI_LLMEngine_Executor* executor, HIAI_LM_Buffer* configBuffer, HIAI_LM_Buffer* loraBuffer,
    HIAI_LM_Buffer* quantBuffer);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SavePromptKVCache(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_PromptKVCache* promptKVCache);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_LoadPromptKVCache(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_PromptKVCache* promptKVCache);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_GetSentDataLen(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, uint32_t* len);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_GetSentData(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, char* sendData, uint32_t len);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SetSparseParam(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, const char* sparseParam);

typedef enum {
    HIAI_LLMEngine_InferPerf_UNSET = 0,
    HIAI_LLMEngine_InferPerf_LOW,
    HIAI_LLMEngine_InferPerf_MIDDLE,
    HIAI_LLMEngine_InferPerf_HIGH,
    HIAI_LLMEngine_InferPerf_EXTREME_HIGH,
} HIAI_LLMEngine_InferPerfMode;

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SetInferencePerfMode(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_InferPerfMode inferPerfMode);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Suspend(
    HIAI_LLMEngine_Executor* executor);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_Resume(
    HIAI_LLMEngine_Executor* executor);

typedef struct HIAI_LLMEngine_ResourceOption HIAI_LLMEngine_ResourceOption;
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_FreeResource(
    HIAI_LLMEngine_Executor* executor, const HIAI_LLMEngine_ResourceOption* resourceOption);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_RestoreResource(
    HIAI_LLMEngine_Executor* executor, const HIAI_LLMEngine_ResourceOption* resourceOption);

HIAI_LLMEngine_ResourceOption* HIAI_LLMEngine_ResourceOption_Create(void);
void HIAI_LLMEngine_ResourceOption_Destroy(HIAI_LLMEngine_ResourceOption** resourceOption);

HIAI_LLMEngine_Status HIAI_LLMEngine_ResourceOption_SetInitOption(
    HIAI_LLMEngine_ResourceOption* resourceOption, const HIAI_LLMEngine_InitOption* initOption);

HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MLLMGenerate(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_Prompt* prompt);
 
HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MLLMGenerateAsync(
    HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_Prompt* prompt);
#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_EXTEND_H
