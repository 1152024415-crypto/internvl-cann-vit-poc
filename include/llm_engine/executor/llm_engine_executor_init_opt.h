/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_EXECUTOR_INIT_OPT_H
#define LLM_ENGINE_EXECUTOR_INIT_OPT_H

#include "common/lm_engine_model_info.h"
#include "executor/llm_engine_executor_init_opt_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HIAI_LLMENGINE_SPM_TOKENIZER = 0,
    HIAI_LLMENGINE_HMB_TOKENIZER = 1,
    HIAI_LLMENGINE_XLMROBERTA_TOKENIZER = 2,
    HIAI_LLMENGINE_QWEN_TOKENIZER = 4,
    HIAI_LLMENGINE_GLM_TOKENIZER = 6,
    HIAI_LLMENGINE_QWEN3_TOKENIZER = 7,
    HIAI_LLMENGINE_UNDEFINED_TOKENIZER = 255,
} HIAI_LLMEngine_TokenizerType;

typedef enum {
    HIAI_LLMENGINE_LLAMA2_MODEL = 0,
    HIAI_LLMENGINE_SMOLVLM_VIT_MODEL = 1,
    HIAI_LLMENGINE_DETACH_PROJ_VIT_MODEL = 2,
    HIAI_LLMENGINE_XIAOYI_VISION_ENCODER_MODEL = 3,
    HIAI_LLMENGINE_XIAOYI_VISION_ADAPTOR_TARGET_MODEL = 4,
    HIAI_LLMENGINE_XIAOYI_VISION_ADAPTOR_DRAFT_MODEL = 5,
    HIAI_LLMENGINE_XIAOYI_AUDIO_ADAPTOR_TRAGET_MODEL = 6,
    HIAI_LLMENGINE_XIAOYI_AUDIO_ADAPTOR__MODEL = 7,
    HIAI_LLMENGINE_XIAOYI_TTS_LLM_MODEL = 8,
    HIAI_LLMENGINE_XIAOYI_TTS_CFM_MODEL = 9,
    HIAI_LLMENGINE_XIAOYI_TTS_HIFIGAN_MODEL = 10,
    HIAI_LLMENGINE_UNDEFINED_MODEL,
} HIAI_LLMEngine_ModelType;

typedef enum {
    HIAI_LLMENGINE_AUTOREGRESSIVE_INFERTYPE = 0,
    HIAI_LLMENGINE_TREESPEC_INFERTYPE = 1,
    HIAI_LLMENGINE_LLMLINGUA_INFERTYPE = 2,
    HIAI_LLMENGINE_UNDEFINED_INFERTYPE,
} HIAI_LLMEngine_InferType;

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetInferType(
    HIAI_LLMEngine_InitOption* initOption,
    HIAI_LLMEngine_InferType inferType
);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetTokenizer(
    HIAI_LLMEngine_InitOption* initOption,
    HIAI_LLMEngine_TokenizerType tokenizerType,
    const char* tokenizerPath
);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetModel(
    HIAI_LLMEngine_InitOption* initOption,
    HIAI_LLMEngine_ModelType modelType,
    HIAI_LMEngine_ModelInfo* modelInfo
);

HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetSpecModel(
    HIAI_LLMEngine_InitOption* initOption,
    HIAI_LLMEngine_ModelType specModelType,
    HIAI_LMEngine_ModelInfo* specModelInfo
);

// kvCacheSwitchableFlag默认是false
HIAI_LLMEngine_Status HIAI_LLMEngine_InitOption_SetKVCacheSwitchableFlag(
    HIAI_LLMEngine_InitOption* initOption,
    bool kvCacheSwitchableFlag
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_INIT_OPT_H
