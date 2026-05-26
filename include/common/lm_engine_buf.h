/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LM_ENGINE_BUF_H
#define LM_ENGINE_BUF_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HIAI_LMENGINE_DATATYPE_UINT8 = 0,
    HIAI_LMENGINE_DATATYPE_INT8,
    HIAI_LMENGINE_DATATYPE_UINT16,
    HIAI_LMENGINE_DATATYPE_INT16,
    HIAI_LMENGINE_DATATYPE_UINT32,
    HIAI_LMENGINE_DATATYPE_INT32,
    HIAI_LMENGINE_DATATYPE_UINT64,
    HIAI_LMENGINE_DATATYPE_INT64,
    HIAI_LMENGINE_DATATYPE_FP16,
    HIAI_LMENGINE_DATATYPE_FP32,
    HIAI_LMENGINE_DATATYPE_FP64,
} HIAI_LMEngine_DataType;

typedef enum {
    HIAI_LMENGINE_MEMORY_TYPE_NORMAL = 0,
    // zero copy
    HIAI_LMENGINE_MEMORY_TYPE_DIRECT_ACCESS,
    HIAI_LMENGINE_MEMORY_TYPE_UNDEFINED
} HIAI_LMEngine_MemoryType;

typedef struct {
    const uint64_t* shape;
    uint64_t dim;
    HIAI_LMEngine_DataType dataType;
    HIAI_LMEngine_MemoryType memoryType;
} HIAI_LMEngine_Tensor_Buf_Info;

#ifdef __cplusplus
}
#endif

#endif // LM_ENGINE_BUF_H
