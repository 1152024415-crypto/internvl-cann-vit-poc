/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_PROMPT_AUDIO_H
#define LLM_ENGINE_PROMPT_AUDIO_H

#include "common/lm_engine_audio.h"
#include "prompt/llm_engine_prompt_def.h"
#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetAudioBufs(
    HIAI_LLMEngine_Prompt* prompt,
    void* audioBufs[],
    size_t audioBufsSize[],
    uint32_t audioNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetAudioInfos(
    HIAI_LLMEngine_Prompt* prompt,
    HIAI_LMEngine_Audio_Info audioInfos[],
    uint32_t audioNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedAudioBufs(
    HIAI_LLMEngine_Prompt* prompt,
    void* audioEncodedBufs[],
    size_t auidoEncodedBufsSize[],
    uint32_t audioNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedAudioBufInfos(
    HIAI_LLMEngine_Prompt* prompt,
    HIAI_LMEngine_Tensor_Buf_Info tensorBufInfos[],
    uint32_t audioNum
);

HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetTTSText(
    HIAI_LLMEngine_Prompt* prompt,
    const char* text
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_PROMPT_AUDIO_H
