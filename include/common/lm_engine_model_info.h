/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2024-2024. All rights reserved.
 * Description:
 */

#ifndef LM_ENGINE_MODEL_INFO_H
#define LM_ENGINE_MODEL_INFO_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t HIAI_LMEngine_Status;

static const HIAI_LMEngine_Status HIAI_LMEngine_SUCCESS = 0;
static const HIAI_LMEngine_Status HIAI_LMEngine_FAILURE = 1;

typedef struct HIAI_LM_Buffer {
    void* data;
    size_t size;
} HIAI_LM_Buffer;

typedef struct HIAI_LMEngine_ModelInfo HIAI_LMEngine_ModelInfo;

HIAI_LMEngine_ModelInfo* HIAI_LMEngine_ModelInfo_Create(void);

void HIAI_LMEngine_ModelInfo_Destroy(
    HIAI_LMEngine_ModelInfo** modelInfo
);

HIAI_LMEngine_Status HIAI_LMEngine_ModelInfo_SetModelBuffer(
    HIAI_LMEngine_ModelInfo* modelInfo,
    HIAI_LM_Buffer* modelBuffer
);

HIAI_LMEngine_Status HIAI_LMEngine_ModelInfo_SetModelPath(
    HIAI_LMEngine_ModelInfo* modelInfo,
    const char* modelPath
);

HIAI_LMEngine_Status HIAI_LMEngine_ModelInfo_SetWeightDir(
    HIAI_LMEngine_ModelInfo* modelInfo,
    const char* weightDir
);

HIAI_LMEngine_Status HIAI_LMEngine_ModelInfo_SetUserData(
    HIAI_LMEngine_ModelInfo* modelInfo,
    void* userData,
    size_t userDataSize
);

HIAI_LMEngine_Status HIAI_LMEngine_ModelInfo_SetModelType(
    HIAI_LMEngine_ModelInfo* modelInfo,
    size_t modelType
);

HIAI_LMEngine_Status HIAI_LMEngine_ModelInfo_SetPreprocessorConfigPath(
    HIAI_LMEngine_ModelInfo* modelInfo,
    const char* preprocessorConfigPath
);

#ifdef __cplusplus
}
#endif

#endif // LM_ENGINE_MODEL_INFO_H