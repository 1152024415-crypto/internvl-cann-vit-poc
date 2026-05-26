/*
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2025. All rights reserved.
 * Description: llm engine return types
 */

#ifndef LLM_ENGINE_RETURN_TYPES_H
#define LLM_ENGINE_RETURN_TYPES_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t HIAI_LLMEngine_Status;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_SUCCESS = 0;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_FAILURE = 1;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_UNINITIALIZED = 2;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_INVALID_PARAM = 3;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_TIMEOUT = 4;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_UNSUPPORTED = 5;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_MEMORY_EXCEPTION = 6;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_INVALID_API = 7;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_INVALID_POINTER = 8;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_CALC_EXCEPTION = 9;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_FILE_NOT_EXIST = 10;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_COMM_EXCEPTION = 11;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_DATA_OVERFLOW = 12;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_SUSPENDED = 13;
static const HIAI_LLMEngine_Status HIAI_LLMEngine_CONNECT_EXCEPTION = 19;

#ifdef __cplusplus
}
#endif

#endif // LLM_ENGINE_RETURN_TYPES_H
