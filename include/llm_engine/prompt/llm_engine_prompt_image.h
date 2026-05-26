/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_PROMPT_IMAGE_H
#define LLM_ENGINE_PROMPT_IMAGE_H

#include "common/lm_engine_image.h"
#include "prompt/llm_engine_prompt_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetImageBufs(
    HIAI_LLMEngine_Prompt* prompt,
    void* imgBufs[],
    size_t imgBufsSize[],
    uint32_t imgNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetImageInfosV2(
    HIAI_LLMEngine_Prompt* prompt,
    HIAI_LMEngine_Image_Info imgInfos[],
    uint32_t imgNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedImageBufs(
    HIAI_LLMEngine_Prompt* prompt,
    void* imgEncodedBufs[],
    size_t imgEncodedBufsSize[],
    uint32_t imgNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedImageBufInfos(
    HIAI_LLMEngine_Prompt* prompt,
    HIAI_LMEngine_Tensor_Buf_Info tensorBufInfos[],
    uint32_t imgNum
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_PROMPT_IMAGE_H
