/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */

#ifndef LLM_ENGINE_EXECUTOR_RESOURCE_OPT_H
#define LLM_ENGINE_EXECUTOR_RESOURCE_OPT_H

#include "llm_engine_return_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HIAI_LLMEngine_ResourceOption HIAI_LLMEngine_ResourceOption;

HIAI_LLMEngine_ResourceOption* HIAI_LLMEngine_ResourceOption_Create(void);

void HIAI_LLMEngine_ResourceOption_Destroy(
    HIAI_LLMEngine_ResourceOption** resourceOption
);

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_EXECUTOR_RESOURCE_OPT_H
