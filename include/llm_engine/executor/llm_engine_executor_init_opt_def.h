/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_EXECUTOR_INIT_OPT_DEF_H
#define LLM_ENGINE_EXECUTOR_INIT_OPT_DEF_H

#include "llm_engine_return_types.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_InitOption HIAI_LLMEngine_InitOption;

HIAI_LLMEngine_InitOption* HIAI_LLMEngine_InitOption_Create(void);

void HIAI_LLMEngine_InitOption_Destroy(
    HIAI_LLMEngine_InitOption** initOption
);


#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_INIT_OPT_DEF_H
