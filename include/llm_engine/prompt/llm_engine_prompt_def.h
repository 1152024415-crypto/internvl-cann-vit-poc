/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_PROMPT_DEF_H
#define LLM_ENGINE_PROMPT_DEF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_Prompt HIAI_LLMEngine_Prompt;

HIAI_LLMEngine_Prompt* HIAI_LLMEngine_Prompt_Create(void);

void HIAI_LLMEngine_Prompt_Destroy(
    HIAI_LLMEngine_Prompt** prompt
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_PROMPT_DEF_H
