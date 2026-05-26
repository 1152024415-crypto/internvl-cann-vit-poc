/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */
 
#ifndef LM_ENGINE_AUDIO_H
#define LM_ENGINE_AUDIO_H

#include "common/lm_engine_buf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const uint64_t* shape;
    uint64_t dim;
    HIAI_LMEngine_DataType dataType;
    HIAI_LMEngine_MemoryType memoryType;
} HIAI_LMEngine_Audio_Info;

#ifdef __cplusplus
}
#endif

#endif // LM_ENGINE_AUDIO_H
