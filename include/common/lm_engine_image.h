/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description:
 */

#ifndef LM_ENGINE_IMAGE_H
#define LM_ENGINE_IMAGE_H

#include "common/lm_engine_buf.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    /** RGBA8888_U8格式的图片 */
    HIAI_LMENGINE_RGBA8888_U8 = 0,
    /** RGB888_U8格式的图片 */
    HIAI_LMENGINE_RGB888_U8 = 1,
    /** BGR888_U8格式的图片 */
    HIAI_LMENGINE_BGR888_U8 = 2,
    HIAI_LMENGINE_IMAGE_FORMAT_INVALID = 255,
} HIAI_LMEngine_ImageFormat;

typedef struct {
    int64_t channel;
    int64_t height;
    int64_t width;
    HIAI_LMEngine_DataType dataType;
    HIAI_LMEngine_ImageFormat imageFormat;
    HIAI_LMEngine_MemoryType memoryType;
    const uint64_t* shape;
    uint64_t dim;
} HIAI_LMEngine_Image_Info;

#ifdef __cplusplus
}
#endif

#endif // LM_ENGINE_IMAGE_H
