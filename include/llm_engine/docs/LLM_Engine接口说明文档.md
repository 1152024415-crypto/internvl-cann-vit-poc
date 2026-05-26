# 一、简介

## 1.1 目的

本文描述了【HiAI支持大模型】LLM Engine接口规格设计说明。供软件开发以及测试人员参考。

### 1.2 用户特征

内部用户，终端软工；

### 1.3 接口描述基本规范

***1) 提供的接口均是c接口，符合c语言描述规范；***

***2) 兼容性考虑，不允许对外暴露结构体的详细定义；***

***3) 变量描述示例：***

| 变量 | <span id="变量名称">变量名称</span> |
| :--- | :--- |
| 定义 | 定义代码 |
| 说明 | xx |
| 值说明 | xxxxx：代码xx意思

***3) 枚举体描述示例：***

| 枚举 | <span id="枚举名字">枚举名字</span> |
| :--- | :--- |
| 定义 | 定义代码 |
| 说明 | xxx |
| 定义说明 | 每个枚举值的意义说明 |

***4) 函数描述示例：***

| 方法原型 | 函数定义 |
| :--- | :--- |
| 参数说明-入参 | xxx |
| 参数说明-出参 | xxx |
| 返回值类型 | xxxx |
| 返回值说明 | [见ModelInfo结构体说明](#HIAI_LMEngine_ModelInfo) |
| 使用说明 | xxxx |

### 1.4 目录

1) [HIAI_LMEngine_ModelInfo介绍](#目录-HIAI_LMEngine_ModelInfo)
2) [HIAI_LLMEngine_Status介绍](#目录-HIAI_LLMEngine_Status)
3) [HIAI_LLMEngine_InitOption介绍](#目录-HIAI_LLMEngine_InitOption)
4) [HIAI_LLMEngine_ResourceOption介绍](#目录-HIAI_LLMEngine_ResourceOption)
5) [HIAI_LLMEngine_Context介绍](#目录-HIAI_LLMEngine_Context)
6) [HIAI_LLMEngine_Prompt介绍](#目录-HIAI_LLMEngine_Prompt)
7) [HIAI_LLMEngine_PromptKVCache介绍](#目录-HIAI_LLMEngine_PromptKVCache)
8) [HIAI_LLMEngine_Executor介绍](#目录-HIAI_LLMEngine_Executor)

### 1.5 修订记录

- 2024/07/25：初始化文档（刘海旭）
- 2024/07/25 ~ 2024/11/04：补充各个特性、功能接口说明（顾康、刘海旭、陈越、王泽、田存华、赵姝馨、王经纬等）
- 2024/11/05：优化文档结构，补充注意事项（顾康）

# 二、接口规格设计

## 2.1 <span id="目录-HIAI_LMEngine_ModelInfo">HIAI\_LMEngine\_ModelInfo</span>

### 2.1.1 介绍

<span id="HIAI_LMEngine_ModelInfo介绍">模型资源信息，头文件：lm_engine/common/lm_engine_model_info.h</span>

### 2.1.2 变量和结构体

| 变量 | <span id="HIAI_LMEngine_Status">HIAI_LMEngine_Status</span> |
| :--- | :--- |
| 定义 | typedef uint32_t HIAI_LMEngine_Status |
| 说明 | 返回值 |
| 值说明 | HIAI_LMEngine_SUCCESS：执行成功 |
| 值说明 | HIAI_LMEngine_FAILURE：执行失败 |

| 结构体 | <span id="HIAI_LM_Buffer">HIAI_LM_Buffer</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LM_Buffer {<br>` `void* data;<br>` `size_t size;<br>} HIAI_LM_Buffer; |
| 说明 | 返回值 |
| 成员说明 | void* data：内存中模型buffer的首地址 |
| 成员说明 | size_t size：模型buffer长度 |

| 结构体 | <span id="HIAI_LMEngine_ModelInfo">HIAI_LMEngine_ModelInfo</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LMEngine_ModelInfo HIAI_LMEngine_ModelInfo; |
| 说明 | 模型资源的管理句柄，内部使用，外部不可见具体定义，仅持有使用 |
| 成员说明 | NA |

### 2.1.3 句柄管理

| 方法原型 | HIAI\_LMEngine\_ModelInfo\*  HIAI\_LMEngine\_ModelInfo\_Create(void) |
| :--- | :--- |
| 参数说明-入参 | NA |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_ModelInfo\* |
| 返回值说明 | [见ModelInfo结构体说明](#HIAI_LMEngine_ModelInfo) |
| 使用说明 | ModelInfo句柄生成，用户持有该句柄配合功能函数进行操作 |

| 方法原型 | void HIAI\_LMEngine\_ModelInfo\_Destroy(HIAI\_LMEngine\_ModelInfo  \*\* modelInfo) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LMEngine\_ModelInfo\*\* modelInfo：句柄的二级指针 |
| 参数说明-出参 | NA |
| 返回值类型 | void |
| 返回值说明 | NA |
| 使用说明 | 释放modelInfo句柄资源 |

### 2.1.4 功能类接口

| 方法原型 | HIAI\_LMEngine\_Status  HIAI\_LMEngine\_ModelInfo\_SetModelBuffer(HIAI\_LMEngine\_ModelInfo\* modelInfo,  HIAI\_LM\_Buffer\* modelBuffer) |
| :--- | :--- |
| 参数说明-入参 |  HIAI\_LMEngine\_ModelInfo\*  modelInfo：句柄指针 |
| 参数说明-入参 |  HIAI\_LM\_Buffer\*  modelBuffer：[见HIAI_LM_Buffer定义](#HIAI_LM_Buffer) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LMEngine\_Status说明](#HIAI_LMEngine_Status) |
| 说明 |  向modleInfo句柄内写入模型Buffer信息，buffer中的内存由用户自己申请，并管理，在模型加载完成之前请勿释放； |

| 方法原型 |  HIAI\_LMEngine\_Status  HIAI\_LMEngine\_ModelInfo\_SetModelPath(HIAI\_LMEngine\_ModelInfo\* modelInfo, const  char\* modelPath) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LMEngine\_ModelInfo\*\*  modelInfo：句柄指针 |
| 参数说明-入参 | const char\*  modelPath模型文件路径：模型文件的路径 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LMEngine\_Status说明](#HIAI_LMEngine_Status) |
| 说明 | 向modleInfo句柄内写入模型路径信息 |

| 方法原型 | HIAI\_LMEngine\_Status  HIAI\_LMEngine\_ModelInfo\_SetWeightDir(HIAI\_LMEngine\_ModelInfo\* modelInfo,  const char\* weightDir) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LMEngine\_ModelInfo\*\*  modelInfo：句柄指针 |
| 参数说明-入参 | const char\*  weightDir权重文件路径：模型权重路径 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LMEngine\_Status说明](#HIAI_LMEngine_Status) |
| 说明 |  向modleInfo句柄内写入权重路径信息 |

## 2.2 <span id="目录-HIAI_LLMEngine_Status">HIAI_LLMEngine_Status</span>

### 2.2.1 介绍

LLM所有返回码，头文件：lm_engine/llm_engine/llm_engine_return_types.h

### 2.2.2 变量和结构体

| 变量 | <span id="HIAI_LLMEngine_Status">HIAI_LLMEngine_Status</span> |
| :--- | :--- |
| 定义 | typedef uint32_t HIAI_LLMEngine_Status |
| 说明 | LLM相关功能函数返回码 |
| 值说明 | HIAI_LLMEngine_SUCCESS：运行正常 |
| 值说明 | HIAI_LLMEngine_FAILURE：内部错误 |
| 值说明 | HIAI_LLMEngine_UNINITIALIZED：模型关键未初始化 |
| 值说明 | HIAI_LLMEngine_INVALID_PARAM：输入参数非法 |
| 值说明 | HIAI_LLMEngine_TIMEOUT：执行任务超时返回 |
| 值说明 | HIAI_LLMEngine_UNSUPPORTED：目前不支持 |
| 值说明 | HIAI_LLMEngine_MEMORY_EXCEPTION：内存异常 |
| 值说明 | HIAI_LLMEngine_INVALID_API：设备不支持此API |
| 值说明 | HIAI_LLMEngine_INVALID_POINTER：非法指针 |
| 值说明 | HIAI_LLMEngine_CALC_EXCEPTION：计算异常 |
| 值说明 | HIAI_LLMEngine_FILE_NOT_EXIST：文件不存在 |
| 值说明 | HIAI_LLMEngine_COMM_EXCEPTION：运行失败 |
| 值说明 | HIAI_LLMEngine_DATA_OVERFLOW：数据溢出 |
| 值说明 | HIAI_LLMEngine_CONNECT_EXCEPTION：连接异常 |

## 2.3 <span id="目录-HIAI_LLMEngine_InitOption">HIAI\_LLMEngine\_InitOption</span>

### 2.3.1 介绍

<span id="HIAI_LLMEngine_InitOption介绍">LLM通用初始化资源信息，头文件：lm_engine/llm_engine/llm_engine_executor.h</span>

### 2.3.2 变量和结构体

| 结构体 | <span id="HIAI_LLMEngine_InitOption">HIAI_LLMEngine_InitOption</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LLMEngine_InitOption HIAI_LLMEngine_InitOption; |
| 说明 | llm资源信息句柄，内部使用，外部不可见具体定义，仅持有使用 |
| 成员说明 | NA |

| 枚举 | <span id="HIAI_LLMEngine_TokenizerType">HIAI_LLMEngine_TokenizerType</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMENGINE_SPM_TOKENIZER = 0,<br>` `HIAI_LLMENGINE_HMB_TOKENIZER = 1,<br>` `HIAI_LLMENGINE_XLMROBERTA_TOKENIZER = 2,<br>` `HIAI_LLMENGINE_UNDEFINED_TOKENIZER,<br>} HIAI_LLMEngine_TokenizerType; |
| 说明 | llm所采用的tokenizer的类型 |
| 定义说明 | HIAI_LLMENGINE_SPM_TOKENIZER：sentencepiece |
| 定义说明 | HIAI_LLMENGINE_HMB_TOKENIZER：蜂鸟模型定制 |
| 定义说明 | HIAI_LLMENGINE_XLMROBERTA_TOKENIZER：prompt压缩模型使用的 |
| 定义说明 | HIAI_LLMENGINE_UNDEFINED_TOKENIZER：未知tokenizer |

| 枚举 | <span id="HIAI_LLMEngine_ModelType">HIAI_LLMEngine_ModelType</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMENGINE_LLAMA2_MODEL = 0,<br>` `HIAI_LLMENGINE_UNDEFINED_MODEL,<br>} HIAI_LLMEngine_ModelType; |
| 说明 | llm所采用的模型类型 |
| 定义说明 | HIAI_LLMENGINE_LLAMA2_MODEL：llama类模型 |
| 定义说明 | HIAI_LLMENGINE_UNDEFINED_MODEL：未知模型 |

| 枚举 | <span id="HIAI_LLMEngine_InferType">HIAI_LLMEngine_InferType</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMENGINE_AUTOREGRESSIVE_INFERTYPE = 0,<br>` `HIAI_LLMENGINE_TREESPEC_INFERTYPE = 1,<br>` `HIAI_LLMENGINE_LLMLINGUA_INFERTYPE = 2,<br>` `HIAI_LLMENGINE_UNDEFINED_INFERTYPE,<br>} HIAI_LLMEngine_InferType;|
| 说明 | llm推理时采用的推理方式 |
| 定义说明 | HIAI_LLMENGINE_AUTOREGRESSIVE_INFERTYPE：自回归推理 |
| 定义说明 | HIAI_LLMENGINE_TREESPEC_INFERTYPE：树型投机推理 |
| 定义说明 | HIAI_LLMENGINE_LLMLINGUA_INFERTYPE：模型压缩使用 |
| 定义说明 | HIAI_LLMENGINE_UNDEFINED_INFERTYPE：未知推理类型 |

### 2.3.3 句柄管理

| 方法原型 | HIAI\_LLMEngine\_InitOption\*  HIAI\_LLMEngine\_InitOption\_Create(void) |
| :--- | :--- |
| 参数说明-入参 | NA | 
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_InitOption\*|
| 返回值说明 |  [见HIAI_LLMEngine_InitOption说明](#HIAI_LLMEngine_InitOption)  |
| 说明 |  返回一个用户持有的InitOption指针，用户使用他在配合功能函数进行操作，内存由内部管理 |

| 方法原型 | void  HIAI\_LLMEngine\_InitOption\_Destroy(HIAI\_LLMEngine\_InitOption\*\* initOption) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_InitOption  \*\* initOption：句柄的二级指针 |
| 参数说明-出参 | NA |
| 返回值类型 | void |
| 返回值说明 | NA |
| 说明 |  释放initOption句柄资源 |

### 2.3.4  功能类接口

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_InitOption\_SetInferType( HIAI\_LLMEngine\_InitOption\*  initOption, HIAI\_LLMEngine\_InferType  inferType) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_InitOption\*  initOption：句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_InferType  inferType：[见HIAI_LLMEngine_InferType说明](#HIAI_LLMEngine_InferType)  |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 向initOption句柄内写入推理类型 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_InitOption\_SetTokenizer( HIAI\_LLMEngine\_InitOption\*  initOption, HIAI\_LLMEngine\_TokenizerType  tokenizerType, const char\*  tokenizerPath) |
| :--- | :--- |
|  参数说明-入参  |  HIAI\_LLMEngine\_InitOption\*  initOption：句柄指针  |
|  参数说明-入参  |  HIAI\_LLMEngine\_TokenizerType  tokenizerType：[见HIAI_LLMEngine_TokenizerType说明](#HIAI_LLMEngine_TokenizerType)  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 向initOption句柄内写入模型词表信息 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_InitOption\_SetModel( HIAI\_LLMEngine\_InitOption\*  initOption, HIAI\_LLMEngine\_ModelType  modelType, HIAI\_LMEngine\_ModelInfo\*  modelInfo) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_InitOption\*  initOption：句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_ModelType  modelType：[见HIAI_LLMEngine_ModelType说明](#HIAI_LLMEngine_ModelType) |
| 参数说明-入参 | HIAI_LMEngine_ModelInfo* modelInfo：[见HIAI_LMEngine_ModelInfo介绍](#HIAI_LMEngine_ModelInfo介绍) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 向initOption句柄内写入模型信息 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_InitOption\_SetSpecModel( HIAI\_LLMEngine\_InitOption\*  initOption, HIAI\_LLMEngine\_ModelType  modelType, HIAI\_LMEngine\_ModelInfo\*  modelInfo) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_InitOption\*  initOption：句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_ModelType  modelType：[见HIAI_LLMEngine_ModelType说明](#HIAI_LLMEngine_ModelType) |
| 参数说明-入参 | HIAI_LMEngine_ModelInfo* modelInfo：[见lm_engine_model_info.h说明](#lm_engine_model_info.h) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 向initOption句柄内写入投机推理小模型信息 |

## 2.4 <span id="目录-HIAI_LLMEngine_ResourceOption">HIAI\_LLMEngine\_ResourceOption</span>

### 2.4.1 介绍

资源管理类选项，当前用在大模型暂停恢复功能中，表示随长暂停恢复需要加载的资源，头文件：lm_engine/llm_engine/llm_engine_executor.h

### 2.4.2 变量和结构体

| 结构体 | <span id="HIAI_LLMEngine_ResourceOption">HIAI_LLMEngine_ResourceOption</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LLMEngine_ResourceOption HIAI_LLMEngine_ResourceOption; |
| 说明 | llm resource资源选项，用于在暂停的时候卸载和加载资源 |
| 成员说明 | NA |

### 2.4.3 句柄管理

| 方法原型 | HIAI\_LLMEngine\_ResourceOption  \* HIAI\_LLMEngine\_ResourceOption\_Create(void) |
| :--- | :--- |
| 参数说明-入参 | NA |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_ResourceOption  \* |
| 返回值说明 | [见HIAI_LLMEngine_ResourceOption说明](#HIAI_LLMEngine_ResourceOption)|
| 说明 | ResourceOption句柄生成，返回一个句柄给用户，用户通过句柄配合功能函数使用 |

| 方法原型 | void  HIAI\_LLMEngine\_ResourceOption\_Destroy(HIAI\_LLMEngine\_ResourceOption \*\* resourceOption) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_ResourceOption  \*\* resourceOption：句柄的二级指针 |
| 参数说明-出参 | NA |
| 返回值类型 | void |
| 返回值说明 | NA |
| 说明 | 释放ResourceOption句柄资源 |

### 2.4.4 功能类接口设计

| 方法原型 | void  HIAI\_LLMEngine\_ResourceOption\_SetInitOption(HIAI\_LLMEngine\_ResourceOption\* resourceOption, const HIAI_LLMEngine_InitOption* initOption) |
| :--- | :--- |
| 参数说明-入参 |  HIAI_LLMEngine_ResourceOption* resourceOption：句柄指针|
| 参数说明-入参 |  const HIAI_LLMEngine_InitOption* initOption句柄：InitOption指针，[见HIAI_LLMEngine_InitOption介绍](#HIAI_LLMEngine_InitOption介绍) |
| 参数说明-出参 | NA |
| 返回值类型 | void |
| 返回值说明 | NA |
| 说明 | 向resourceOption中设置initOption |

## 2.5 <span id="目录-HIAI_LLMEngine_Context">HIAI\_LLMEngine\_Context</span>

### 2.5.1 介绍

大模型推理时的上下文管理，头文件：lm_engine/llm_engine/llm_engine_context.h
***使用注意：***
***1）context不支持多线程操作，如果需要，请每个线程创建自己单独的context；***
***2）context指针在executor中被使用，请确认executor不在运行后，在销毁context；***

### 2.5.2 变量和结构体

| 结构体 | <span id="HIAI_LLMEngine_Context">HIAI_LLMEngine_Context</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LLMEngine_Context HIAI_LLMEngine_Context; |
| 说明 | llm 模型运行的上下文管理 |
| 成员说明 | NA |

| 结构体 | <span id="callbackFuncType">callbackFuncType</span> |
| :--- | :--- |
| 定义 | typedef void(*callbackFuncType)(const HIAI_LLMEngine_Context\*); |
| 说明 | llm 回调函数定义，<br>***注意：该函数没有返回值，内部传入的ctx的指针地址和外部用户持有的相同，可以利用地址来区分不同的ctx回调；*** |
| 成员说明 | NA |

| 变量 | <span id="HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN">HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN</span> |
| :--- | :--- |
| 定义 | #define HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN 10 |
| 说明 | 停止词字符串的最大个数 |
| 成员说明 | NA |

| 变量 | <span id="HIAI_LLM_ENGINE_COMPRESS_FORCE_TOKENS_MAX_LEN">HIAI_LLM_ENGINE_COMPRESS_FORCE_TOKENS_MAX_LEN</span> |
| :--- | :--- |
| 定义 | #define HIAI_LLM_ENGINE_COMPRESS_FORCE_TOKENS_MAX_LEN 100 |
| 说明 | prompt压缩保留token的最大个数 |
| 成员说明 | NA |

| 枚举 | <span id="HIAI_LLMEngine_SpecTreeType">HIAI_LLMEngine_SpecTreeType</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMENGINE_STATIC_SPEC_TREE = 0,<br>` `HIAI_LLMENGINE_DYNAMIC_SPEC_TREE = 1,<br>` `HIAI_LLMENGINE_SPEC_TREE_TYPE_INVALID<br>} HIAI_LLMEngine_SpecTreeType;|
| 说明 | 投机树的类型 |
| 定义说明 | HIAI_LLMENGINE_STATIC_SPEC_TREE：静态树 |
| 定义说明 | HIAI_LLMENGINE_DYNAMIC_SPEC_TREE：动态树 |
| 定义说明 | HIAI_LLMENGINE_SPEC_TREE_TYPE_INVALID：未知类型 |

| 枚举 | <span id="HIAI_LLMEngine_DynGammaStrategy">HIAI_LLMEngine_DynGammaStrategy</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMENGINE_GAMMA_DISABLE = 0,<br>` `HIAI_LLMENGINE_GAMMA_EMA_RISE_EMA_FALL = 1,<br>` `HIAI_LLMENGINE_GAMMA_FORCE_RISE_FORCE_FALL = 2,<br>` `HIAI_LLMENGINE_GAMMA_FORCE_RISE_STAIRWAY_FALL = 3,<br>` `HIAI_LLMENGINE_GAMMA_FORCE_RISE_EMA_FALL = 4,<br>` `HIAI_LLMENGINE_GAMMA_STRATEGY_INVALID<br>} HIAI_LLMEngine_DynGammaStrategy;|
| 说明 | 动态gamma策略类型 |
| 定义说明 | HIAI_LLMENGINE_GAMMA_DISABLE：不使能动态gamma功能 |
| 定义说明 | HIAI_LLMENGINE_GAMMA_EMA_RISE_EMA_FALL：指数移动平均上升指数移动平均下降 |
| 定义说明 | HIAI_LLMENGINE_GAMMA_FORCE_RISE_FORCE_FALL：gamma值固定设置为上一轮接受token数加1 |
| 定义说明 | HIAI_LLMENGINE_GAMMA_FORCE_RISE_STAIRWAY_FALL：gamma值上升时设置为上一轮接受token数加1，下降时每次降1 |
| 定义说明 | HIAI_LLMENGINE_GAMMA_FORCE_RISE_EMA_FALL：gamma值上升时设置为上一轮接受token数加1，下降时采用指数平均下降 |
| 定义说明 | HIAI_LLMENGINE_GAMMA_STRATEGY_INVALID：未知类型 |

| 枚举 | <span id="HIAI_LLMEngine_SpecType">HIAI_LLMEngine_SpecType</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMENGINE_TREE_SPECTYPE = 0,<br>` `HIAI_LLMENGINE_TEXT_SPECTYPE = 1,<br>` `HIAI_LLMENGINE_UNDEFINED_SPECTYPE<br>} HIAI_LLMEngine_SpecType;|
| 说明 | 投机推理的类型 |
| 定义说明 | HIAI_LLMENGINE_TREE_SPECTYPE：树形投机 |
| 定义说明 | HIAI_LLMENGINE_TEXT_SPECTYPE：文本投机 |
| 定义说明 | HIAI_LLMENGINE_UNDEFINED_SPECTYPE：未知类型 |

### 2.5.3 句柄管理

| 方法原型 | HIAI\_LLMEngine\_Context  \* HIAI\_LLMEngine\_Context\_Create(void) |
| :--- | :--- |
| 参数说明-入参 | NA |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Context  \* |
| 返回值说明 | [见HIAI_LLMEngine_Context说明](#HIAI_LLMEngine_Context) |
| 说明 | Context句柄生成，返回一个句柄给用户，需要配合功能函数使用 |

| 方法原型 | void HIAI\_LLMEngine\_Context\_Destroy(HIAI\_LLMEngine\_Context\*\*  ctx) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*\*  ctx：句柄的二级指针 |
| 参数说明-出参 | NA |
| 返回值类型 | void |
| 返回值说明 | NA |
| 说明 | 释放Context句柄 |

### 2.5.4  基本推理控制类接口设计

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_TerminateOnce( HIAI\_LLMEngine\_Context\*  ctx)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 终止一次模型推理；<br>***注意：这里只是配置一下停止标志，异步的情况下需要配合[回调函数](#callbackfunc介绍)判断是否已经结束。*** |

### 2.5.5  基本推理参数设置类接口设计

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetCallbackFreq( HIAI\_LLMEngine\_Context\*  ctx, uint32\_t callbackFreq) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | uint32\_t callbackFreq：回调函数执行频率参数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 |  [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置模型推理时的回调函数执行频率，可选参数： <br>1. callbackFreq=0：默认值，推理过程中不做回调，整个推理完成后才会生成输出 <br>2. callbackFreq>0：当推理得到的token数满足该参数时，执行一次回调，输出当前推理生成的字符串 <br>3. callbackFreq<0：非法值，不支持传入 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetCallbackFreq( HIAI\_LLMEngine\_Context\*  ctx, uint32\_t\* callbackFreq) |
| :--- | :--- |
| 参数说明-入参 |  HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  int\* callbackFreq：回调函数的使用频率 |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 |  [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取模型推理的回调函数执行频率  |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetSeed( HIAI\_LLMEngine\_Context\*  ctx, uint32\_t seed) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | uint32\_t seed：采样时用到的随机数种子 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 |  [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置模型采样时用的随机数种子，默认值99；<br>***注意：如果每次推理不设置，仅打开采样选项，那么每次生成的随机数序列是一样的，采样结果也没区别；*** |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetSeed( const HIAI\_LLMEngine\_Context\*  ctx, uint32\_t\* seed) |
| :--- | :--- |
| 参数说明-入参 |  const HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  uint32\_t\* seed：当前随机数种子值 |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 |  [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取模型推理的随机数种子 |

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetDoSampleFlag( HIAI\_LLMEngine\_Context\*  ctx, bool f)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针  |
| 参数说明-入参 | bool f：是否进行输出采样的标记位  |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 |  [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置是否进行输出采样<br>***注意：如果不设置，默认不进行采样，那么topk、topp等采样的参数就算是配置了也不会生效，只会输出每轮概率最大的token。*** |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetDoSampleFlag( const HIAI\_LLMEngine\_Context\*  ctx, bool\* f) |
| :--- | :--- |
| 参数说明-入参 |  const HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  bool\* f：是否进行输出采样的标记位 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取输出采样标记位 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetTopK( HIAI\_LLMEngine\_Context\*  ctx, uint64\_t k) |
| :-- | :-- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | uint64\_t k topK：k值，具体含义见说明 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置输出采样topK参数，默认值为100，在采样时，选取的前k个概率最大的，可选参数：<br> 1. topK<100：不生效的参数，Engine内部保留minKeep字段，为保证输出多样性，建议sampling阶段最少保留100个备选token；<br> 2. topK>100：保留的备选token数量在min(topK, logits.size())中；  |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetTopK( const HIAI\_LLMEngine\_Context\*  ctx, uint64\_t\* k) |
| :--- | :--- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 | uint64\_t\* k：当前采样topK参数 |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取输出采样topK参数 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetTopP(HIAI\_LLMEngine\_Context\* ctx, float p) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | float p：采样参数topP |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置输出采样参数topP，p值代表选取前几个概率加起来大于p的token；<br>1. >0切<=1是正常值；<br>2. <=0是非法值；  |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetTopP( const HIAI\_LLMEngine\_Context\*  ctx, float\* p) |
| :--- | :--- |
| 参数说明-入参 |  const HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  float\* p：当前采样参数topP |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取输出采样参数topP |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetTemperature(HIAI\_LLMEngine\_Context\* ctx, float t)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | float t：输入的temp值 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置输出采样参数temperature，用于描述缩放概率数值，默认值1.0，可选参数：<br> 1. >1或者<1：正常的值，描述缩放倍率； <br> 2. =0：非法值；|

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetTemperature( const HIAI\_LLMEngine\_Context\*  ctx, float\* t) |
| :--- | :--- |
| 参数说明-入参 |  const HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  float\* t：当前采样参数temperature  |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取输出采样参数temperature |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetRepetitionPenalty(HIAI\_LLMEngine\_Context\* ctx,  float rp) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | float rp：token重复生成惩罚参数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置输出采样参数的重复惩罚参数，如果大于1，则表示输出的重复率越高，则该token作为输出的可能性越小；反之表示输出的重复率越高，则该token作为输出的可能性也越高；|

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetRepetitionPenalty( const HIAI\_LLMEngine\_Context\*  ctx, float\* rp)  |
| :--- | :--- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  float\* rp：当前采样参数repetition penalty |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取输出采样参数repetition penalty |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_SetMaxGenTokens( HIAI\_LLMEngine\_Context\* ctx, uint64\_t m)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-入参 | uint64\_t m：最大生成token数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置最大输出token个数，默认值16；<br>***注意：这里指的是新生成的token上限，一般模型还会有一个输入输出中token的限制，在配置文件中，生成token上限会受着两个同时控制；*** |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetMaxGenTokens( const HIAI\_LLMEngine\_Context\*  ctx, uint32\_t\* m)  |
| :--- | :--- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\*  ctx Context句柄 |
| 参数说明-出参 | uint32\_t\* m：最大输出token数 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取最大输出token数 |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_SetStopSeq( HIAI\_LLMEngine\_Context\* ctx, char\*\* stopSeq, uint32\_t stopSeqLen)  |
| :---- | :---- |
| 参数说明-入参  | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-入参  | char\*\* stopSeq：截止符数组 |
| 参数说明-入参  | uint32\_t stopSeqLen：截止符个数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置模型推理时的截止符，可以有多个截止符，每个截止符都是一个char\*类型的字符串，[最大值见HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN](#HIAI_LLM_ENGINE_STOP_SEQ_MAX_LEN) ；<br>***注意：重复调用该接口的话，截止符不会累加，内部会清空上一次设置的所有截止符***|

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetTotalTimeMs(const  HIAI\_LLMEngine\_Context\* ctx, double\* ms) |
| :---- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 | double\* ms：生成token的总时间 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取整个推理流程的总耗时；<br>***注意：llm engine一次完整的生成结束后会刷新，请在一次完整生成后调用获取，其余时间调用的值不可信；*** |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetPrefillTimeMs(const  HIAI\_LLMEngine\_Context\* ctx, double\* ms)  |
| :--- | :--- |
| 参数说明-入参 |  const HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 |  double\* ms：预填充阶段（首token）的耗时 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取预填充阶段耗时； |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetDecodeTimeMs(const  HIAI\_LLMEngine\_Context\* ctx, double\* ms) |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 | double\* ms：decode阶段（接龙帧）的耗时 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取接龙帧推理的总耗时 |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetInputTokenCount(const  HIAI\_LLMEngine\_Context\* ctx, uint64\_t\*count) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 | uint64\_t\* count：输入token个数 |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取输入的总token个数； |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetOutputTokenCount(const  HIAI\_LLMEngine\_Context\* ctx, uint64\_t\*count)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 | uint64\_t\* count：输出token个数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取输出的总token个数； |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetOneGenerationLen(const HIAI_LLMEngine_Context\* ctx, uint32_t\* len)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 | uint32_t\* len：生成的字符串长度 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取一次回调生成的字符串长度； |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetOneGeneration(const  HIAI\_LLMEngine\_Context\* ctx, char\* generation, uint32\_t len) |
| :--- | :--- |
| 参数说明-入参 |  const HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 |  char\*\* generation：输出字符数组指针 |
| 参数说明-入参 |  uint32\_t len：输出字符个数 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取单次推理得到的字符串；<br>***注意：传入的字符数组由用户自己申请管理，len长度必须是通过GetOneGenerationLen接口获取的；*** |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetAllGenerationLen(const  HIAI\_LLMEngine\_Context\* ctx, uint32\_t\* len)  |
| :--- | :--- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-出参 | uint32\_t\* len：全部推理得到的字符串长度 |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取一次生成完成后生成的字符串长度； |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_GetAllGeneration(const  HIAI\_LLMEngine\_Context\* ctx, char\* generation, uint32\_t len) |
| :---- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx Context句柄 uint32\_t len全部推理得到的字符串长度 |
| 参数说明-出参 |  char\* generation全部推理的输出字符串 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取全部推理的输出字符串；<br>***注意：传入的len必须是由GetAllGenerationLen接口得到。*** |

| 方法原型 | HIAI_LLMEngine_Status HIAI\_LLMEngine\_Context\_GetGenerateStatus(const HIAI\_LLMEngine\_Context* ctx) |
| :---- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx Context句柄 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 调用失败之后，从ctx中回去具体的失败返回码； |

### 2.5.6  回调函数相关接口设计

| 方法原型 |  HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_SetOnSomeTokenGenerateDoneFunc(HIAI\_LLMEngine\_Context\*  ctx, callbackFuncType func) |
| :---- | :----|
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-入参 | callbackFuncType func：[见回调函数定义](#callbackFuncType) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置流式回调函数的接口，这个接口配合SetCallbackFreq使用可以完成，异步的情况下生成几个token就回调一次的能力；<br>***注意：同步接口也会回调该函数。*** |

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetOnAllTokensGenerateDoneFunc(HIAI\_LLMEngine\_Context\*  ctx, callbackFuncType func) |
| :---- | :----|
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-入参 | callbackFuncType func：[见回调函数定义](#callbackFuncType) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置所有token生成完成的回调函数，一般配合异步接口使用；<br>***注意：同步接口也会回调该函数。*** |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Context\_SetOnGenerateAsyncFailed(HIAI\_LLMEngine\_Context\*  ctx, callbackFuncType func)  |
| :---- | :----|
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx：句柄指针 |
| 参数说明-入参 | callbackFuncType func：[见回调函数定义](#callbackFuncType) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置异步推理失败时发生的回调函数；<br>***注意：该函数只有异步推理会回调。*** |

### 2.5.7  投机推理相关接口设计

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetInitTokenLen (HIAI\_LLMEngine\_Context\* ctx, uint32\_t  initTokenlen)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | uint32\_t initTokenLen：初始不覆盖的token数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置initTokenLen值，这个参数源自于streamllm，在kv cache的buffer发生绕会时，会有最早的initTokenLen个token的kv cache被保留，不覆盖； |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetInitTokenLen (HIAI\_LLMEngine\_Context\* ctx, uint32\_t\*  initTokenlen)  |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | uint32\_t\* initTokenLen：当前配置的初始的token个数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取initTokenLen的值 ；|

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetSpecTree(HIAI\_LLMEngine\_Context\* ctx, uint32\_t  gamma, uint32\_t\* topHead, uint32\_t\* topKSpec) |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针  |
| 参数说明-入参 | uint32\_t gamma：投机树的深度 |
| 参数说明-入参 | uint32\_t\*  topHead：长度的列表，第(i)层的点选择前topHead[i]个子节点继续生长； |
| 参数说明-入参 | uint32\_t\*  topKSpec：长度的列表，第(i)层中继续生长的节点，每个节点生成topKSpec[i]子节点；  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置投机推理时的校验类型；|

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetTypicalScaling (HIAI\_LLMEngine\_Context\* ctx,  float typicalScaling)  |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-入参 | float typicalScaling：拒绝采样超参 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置拒绝采样超参typicalScaling |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetTypicalScaling(HIAI\_LLMEngine\_Context\* ctx, float\*  typicalScaling)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 | float\* typicalScaling：需要获取的拒绝采样超参 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取拒绝采样超参typicalScaling |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetDraftThreashold (HIAI\_LLMEngine\_Context\* ctx, float  draftThreashold) |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针  |
| 参数说明-入参 |  float draftThreashold：在校验阶段能接受的小模型token的最小概率  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置小模型的概率阈值；<br>***注意：当token的概率小于等于此阈值时校验不通过。默认值0.05；***  |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetDraftThreashold (HIAI\_LLMEngine\_Context\* ctx, float\*  draftThreashold)  |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 | float\* draftThreashold：需要获取的小模型的概率阈值 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取小模型的概率阈值 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetSampleGreedy (HIAI\_LLMEngine\_Context\* ctx, bool  greedy)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针  |
| 参数说明-入参 | bool greedy：在校验阶段是否贪心采样 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置是否贪心采样；<br>****注意：如果为true，则在token校验通过后，不再继续校验同一父节点下的其他叶子节点对应的token；默认值false ** |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetSampleGreedy (HIAI\_LLMEngine\_Context\* ctx, bool\*  greedy)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针  |
| 参数说明-出参 | bool\* greedy：获取是否贪心采样  |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取是否贪心采样； |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_SetRejectScaling (HIAI\_LLMEngine\_Context\* ctx, bool  rejectScaling) |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：句柄指针  |
| 参数说明-入参 | bool rejectScaling：在校验阶段是否根据当前小模型token的概率，放大下一个token的概率  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置是否动态调整token校验通过率；<br>***注意：rejectScaling在校验阶段是否根据当前小模型token的概率，放大下一个token的概率，以此来增大下一个token校验通过的概率；默认值false；*** |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetRejectScaling (HIAI\_LLMEngine\_Context\* ctx, bool\*  rejectScaling)  |
| :---- | :---- |
| 参数说明-入参  | HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参  | bool\* rejectScaling：获取是否动态调整token校验通过率 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取是否动态调整token校验通过率；|

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Context\_GetDecodeNum (HIAI\_LLMEngine\_Context\* ctx, uint64\_t\*num)  |
| :---- | :---- |
| 参数说明-入参 |  HIAI\_LLMEngine\_Context\*  ctx：句柄指针 |
| 参数说明-出参 |  uint64\_t\* num：输出token个数 |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取推理输出的token个数 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_ClearSpecTree(HIAI_LLMEngine_Context* ctx)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 在设置树之前需要调用该接口将树的列表清空，然后依次进行多棵树的设置 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeType(HIAI_LLMEngine_Context* ctx, uint32_t treeIdx, HIAI_LLMEngine_SpecTreeType type)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t treeIdx`: 树的索引<br>`HIAI_LLMEngine_SpecTreeType type`: 树的类型，[见 HIAI\_LLMEngine\_SpecTreeType 说明](#HIAI_LLMEngine_SpecTreeType) |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 多棵树的设置循环调用该接口，按顺序treeIdx依次传入0、1、2、3…… |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeShape(HIAI_LLMEngine_Context* ctx, uint32_t treeIdx, uint32_t gamma, uint32_t* topHead, uint32_t** topKSpec, uint32_t  topKSpecDim)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t treeIdx`: 树的索引<br>`uint32_t gamma`: 树的深度<br>`uint32_t* topHead`: gamma长度的数组，第(i)层的点选择前topHead[i]个子节点继续生长<br>`uint32_t** topKSpec`: 二维数组[gamma][topKSpecDim]，第(i)层中继续生长的节点，每个节点生成topKSpec[i]子节点uint32_t  topKSpecDim: 二维数组topKSpec第二维的维度 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 用户调用该接口n次进行n棵树的形状配置 |

| 方法原型 | `HIAI_LLMENGINE_C_API_EXPORT HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDynGammaStrategy(HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_DynGammaStrategy strategy);` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`HIAI_LLMEngine_DynGammaStrategy strategy`: gamma选择策略，[见 HIAI_LLMEngine_DynGammaStrategy 说明](#HIAI_LLMEngine_DynGammaStrategy)|
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 设置动态 gamma 策略类型 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDynGammaEmaWindow(HIAI_LLMEngine_Context* ctx, uint32_t emaWindow);` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br> `uint32_t emaWindow`: 动态gamma指数平均窗口长度|
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 设置动态gamma指数移动平均策略的窗口长度 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecTreeConfidenceThresh(HIAI_LLMEngine_Context* ctx,uint32_t treeIdx, float thresh);` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br> `float thresh`: 起草中止的置信度阈值，范围[0,1]，当阈值设置为0时可关闭起草中止功能|
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 设置起草中止功能的置信度阈值 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetSpecType(HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_SpecType specType)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br> `HIAI_LLMEngine_SpecType specType`: 投机推理类型|
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 设置投机类型 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetDraftText(HIAI_LLMEngine_Context* ctx, const char* draftText)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br> `const char* draftText`: 候选草稿内容|
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 设置文本投机的候选草稿内容 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetMatchLen(HIAI_LLMEngine_Context* ctx, uint32_t matchLen)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t matchLen`: 用于搜索草稿的子串长度 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 推荐取值 matchLen=3。<br>使用已经过验证的最近 matchLen 个 token 作为模式匹配的子串，用于在候选草稿中搜索本轮草稿结果。<br>例如当前输出结果为"尊敬的领导", matchLen设置为 3, 则起草阶段使用 "的领导" 三个token在候选草稿中搜索和这三个token匹配的位置，将该位置之后的候选草稿作为起草结果送给大模型校验|

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetWindowSize(HIAI_LLMEngine_Context* ctx, uint32_t windowSize)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t windowSize`: 搜索草稿的窗口大小 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | `HIAI_LLMEngine_SUCCESS`: 成功<br>`HIAI_LLMEngine_FAILURE`: 失败 |
| 说明 | 推荐取值 windowSize=100。<br>如果当前验证位置为 pos, 在 [pos - windowSize, pos + windowSize] 的范围内搜索草稿。<br>例如候选草稿为“尊敬的领异：\n\n祝您和您的家人中秋佳节快乐！\n\n尽管我离开公司，但我相信我们的关系不会因此改番。”，第一轮校验后通过校验“尊敬的领异：\n\n祝”，并由大模型生成“您”，则验证起点pos=18，即“和”所在位置，例如windowSize=3，则搜索范围是 [15, 21]，matchLen=3, 则取出“\n祝您”在“\n祝您和您的家”中搜索匹配位置，最终搜索到的位置为18|
### 2.5.8 Prompt KV Cache

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefixPrompt(HIAI_LLMEngine_Context* ctx, const char* prefixPrompt) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context* ctx: 句柄指针 |
| 参数说明-入参 |  const char* prefixPrompt： 设置的prompt文本|
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  设置prefix prompt |

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Context_UnSetPrefixPrompt(HIAI_LLMEngine_Context* ctx)  |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Context* ctx: 模型执行上下文 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 取消prefix prompt |

### 2.5.9 Prompt压缩

| 方法原型   | HIAI_LLMEngine_Status HIAI\_LLMEngine\_Context\_SetCompressRatio(HIAI\_LLMEngine\_Context* ctx, float ratio)  |
| :---- | :---- |
|  参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：上下文句柄指针 |
|  参数说明-入参 | float ratio：压缩率 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | prompt压缩率设置；<br>***注意：ratio的范围是(0, 1)***  |

| 方法原型 | HIAI_LLMEngine_Status HIAI\_LLMEngine\_Context\_SetAdditionalContext(HIAI\_LLMEngine\_Context* ctx, char* additionalContext)  |
| :---- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：上下文句柄指针 |
| 参数说明-入参 | char* additionalContext：压缩模型执行时的additionalContext |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置压缩模型执行时的additionalContext，QA任务中通常是query； |

| 方法原型 | HIAI_LLMEngine_Status HIAI\_LLMEngine\_Context\_SetForceTokens(HIAI\_LLMEngine\_Context* ctx, char** forceTokens, uint32_t forceTokensLen)  |
| :---- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：上下文句柄指针 |
| 参数说明-入参 | char** forceTokens：压缩时必须保留的单词，以二级指针的形式传入 |
| 参数说明-入参 | int32_t forceTokensLen：压缩时必须保留的单词数 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置压缩模型执行时必须保留的单词，通常是标点符号；|

| 方法原型 | HIAI_LLMEngine_Status HIAI\_LLMEngine\_Context\_GetCompressPromptLen(HIAI\_LLMEngine\_Context* ctx, uint32_t* len)  |
| :---- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：上下文句柄指针 |
| 参数说明-出参 | int32_t* len：获取压缩结果长度 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取压缩之后的提示词长度 |

| 方法原型 | HIAI_LLMEngine_Status HIAI\_LLMEngine\_Context\_GetCompressPrompt(HIAI\_LLMEngine\_Context* ctx, char* compressPrompt, uint32_t len)  |
| :----- | :---- |
| 参数说明-入参 | const HIAI\_LLMEngine\_Context\* ctx：上下文句柄指针 |
| 参数说明-出参 | char* compressPrompt：压缩后提示词的存放地址 |
| 参数说明-入参 | int32_t len：压缩后提示词长度 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 获取压缩之后的提示词，保存在compressPrompt指向的内存中；<br>***注意：长度必须是使用GetCompressPromptLen获取的，并且字符串的内存用户自己申请管理；*** |

## 2.6 <span id="目录-HIAI_LLMEngine_Prompt">HIAI\_LLMEngine\_Prompt</span>

### 2.6.1 介绍

prompt句柄，用于设置不同模态的prompt输入，头文件：lm_engine/llm_engine/llm_engine_executor.h

### 2.6.2 变量和结构体

| 结构体 | <span id="HIAI_LLMEngine_Prompt">HIAI_LLMEngine_Prompt</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LLMEngine_Prompt HIAI_LLMEngine_Prompt; |
| 说明 | llm prompt 句柄 |
| 成员说明 | NA |

### 2.6.2 句柄管理

| 方法原型 |  HIAI_LLMEngine_Prompt* HIAI_LLMEngine_Prompt_Create(void) |
| :---- | :---- |
| 参数说明-入参 | NA |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI_LLMEngine_Prompt*  |
| 返回值说明 |  多模态Prompt指针 |
| 说明  |  多模态模型Prompt创建；<br>***注意：当prompt中只存在text的时候也可以使用这个接口，并且后续有将所有prompt设置都放到这个句柄上来的计划，更加灵活，兼容性好；***  |

| 方法原型 | void HIAI_LLMEngine_Prompt_Destroy(<br>HIAI_LLMEngine_Prompt** prompt)  |
| :--- | :---- |
| 参数说明-入参 | HIAI_LLMEngine_Prompt** prompt：句柄的二级指针 |
| 参数说明-出参 | NA |
| 返回值类型  |  void  |
| 返回值说明  |  NA |
| 说明 | 释放多模态Prompt句柄资源； |

### 2.6.3 功能类接口设计

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetText(HIAI_LLMEngine_Prompt* prompt , const char* text)  |
| :--- | :--- |
| 参数说明-入参 | HIAI_LLMEngine_Prompt* prompt ：句柄指针 |
| 参数说明-入参 | const char* text：文本 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  多模态模型Prompt设置文本 |

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetImageBufs(HIAI_LLMEngine_Prompt* prompt, void* imgBufs[], size_t imgBufsSize[], uint32_t imgNum)   |
| :---- | :---- |
|  参数说明-入参  | HIAI_LLMEngine_Prompt* prompt：句柄指针 |
|  参数说明-入参  | void* imgBufs[]：图像Buffer数组 |
|  参数说明-入参  | uint32_t imgNum：图像个数|
| 参数说明-出参 | NA |
| 返回值类型  |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明  |  多模态模型Prompt设置图像buffer数组 |

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedImageBufs(HIAI_LLMEngine_Prompt* prompt, void* imgEncoderBufs[], size_t imgEncodedBufsSize[], uint32_t imgNum)   |
| :---- | :---- |
| 参数说明-入参 | HIAI_LLMEngine_Prompt* prompt：句柄指针 |
| 参数说明-入参 | void* imgEncoderBufs[]：图像编码后的Buffer数组 |
| 参数说明-入参 | uint32_t imgNum：图像个数 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 多模态模型Prompt设置图像编码后的buffer数组；<br>***注意：这边是为了节省下图像embedding的时间，所以可以直接设置图像embedding后的内容，内部检测到设置这个之后优先会使用这个值，即便设置图像也没有用；*** |

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Prompt_SetEncodedAudioBufs(HIAI_LLMEngine_Prompt* prompt, void* audioEncodedBufs[], size_t audioEncodedBufsSize[], uint32_t audioNum)   |
| :---- | :---- |
| 参数说明-入参 | HIAI_LLMEngine_Prompt* prompt：句柄指针 |
| 参数说明-入参 | void* audioEncodedBufs[]：语音编码后的Buffer数组 |
| 参数说明-入参 | uint32_t audioEncodedBufsSize：语音编码后的Buffer数组每个Buffer的大小 |
| 参数说明-入参 | uint32_t audioNum：语言编码后的audioEncodedBufs数组维度 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 多模态模型Prompt设置语音编码后的buffer数组；<br>***注意：此处是S2TT场景给模型设置语音embedding后的内容 当前只支持audioNum不超过2, audioEncodedBufs按先大模型后小模型排布*** |

## 2.7 <span id="目录-HIAI_LLMEngine_PromptKVCache">HIAI\_LLMEngine\_PromptKVCache</span>

### 2.7.1 介绍

系统自带的prompt模板可以进行临时的存储，以便后续遇到同场景可以避免重复推理所造成的性能损失，用户可以设置这些prompt，支持单句和多句；头文件：lm_engine/llm_engine/llm_engine_executor.h

### 2.7.2 变量和结构体

| 结构体 | <span id="HIAI_LLMEngine_PromptKVCache">HIAI_LLMEngine_PromptKVCache</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LLMEngine_PromptKVCache HIAI_LLMEngine_PromptKVCache; |
| 说明 | llm prompt kv cache句柄 |
| 成员说明 | NA |

### 2.7.2 句柄管理

| 方法原型 | HIAI_LLMEngine_PromptKVCache* HIAI_LLMEngine_PromptKVCache_Create()  |
| :--- | :--- |
| 参数说明-入参 | NA  |
| 参数说明-出参 | NA  |
| 返回值类型 | HIAI\_LLMEngine\_PromptKVCache*  |
| 返回值说明 | PromptKVCache句柄类型的指针  |
| 说明 | 创建PromptKVCache句柄 |

| 方法原型 | HIAI_LLMEngine_PromptKVCache* HIAI_LLMEngine_PromptKVCache_CreateFromBuffer(void* data, uint64_t size)  |
| :---- | :---- |
| 参数说明-入参 |  void* data：指向prefix prompt KV cache内存的首地址； |
| 参数说明-入参 |  uint64_t size：data指向的内存申请的空间大小； |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_PromptKVCache* |
| 返回值说明 | PromptKVCache句柄类型的指针 |
| 说明  |  从指定buffer创建PromptKVCache句柄 |

| 方法原型 | void HIAI_LLMEngine_PromptKVCache_Destory( HIAI_LLMEngine_PromptKVCache** promptKVCache)   |
| :---- | :---- |
| 参数说明-入参 |  HIAI_LLMEngine_PromptKVCache** promptKVCache： PromptKVCache句柄的二级指针；  |
| 参数说明-出参 | NA |
| 返回值类型 | void |
| 返回值说明 | NA |
| 说明 |  释放PromptKVCache句柄，若句柄内kv cache内存由海思创建，释放该内存 |

### 2.7.3 功能类接口设计

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_PromptKVCache_GetData(HIAI_LLMEngine_PromptKVCache* promptKVCache, void** data, uint64_t* size)  |
| :--- | :---- |
| 参数说明-入参 | HIAI_LLMEngine_PromptKVCache* promptKVCache：句柄指针 |
| 参数说明-出参 | void** data：cache data的二级指针 |
| 参数说明-出参 | uint64_t* size：data大小 |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  获取promptKVCache句柄中data的首地址和大小 |

## 2.8 <span id="目录-HIAI_LLMEngine_Executor">HIAI\_LLMEngine\_Executor</span>

### 2.8.1 介绍

HIAI\_LLMEngine\_Executor作为对外提供大模型推理功能的执行器句柄，需要通过Create方法创建，并通过Destroy方法销毁，用户持有该句柄针对某个大模型进行一次推理生成任务，头文件：lm_engine/llm_engine/llm_engine_executor.h

***使用注意：***
***1）用户在执行LLMEngine推理时应按照正确的流程调用，否则容易发生Crash，具体流程如下：***
***① 调用句柄的创建接口：HIAI\_LLMEngine\_Executor\_Create，HIAI\_LLMEngine\_Context\_Create，HIAI\_LLMEngine\_InitOption\_Create等；***
***② 调用资源初始化接口：HIAI\_LLMEngine\_Executor\_Init；***
***③ 调用参数设置类接口；***
***④ 调用推理接口：HIAI\_LLMEngine\_Executor\_Generate或HIAI\_LLMEngine\_Executor\_GenerateAsync；***
***⑤ 调用资源卸载接口：HIAI\_LLMEngine\_Executor\_DeInit；***
***⑥ 释放句柄资源：HIAI\_LLMEngine\_Executor\_Destroy以及HIAI\_LLMEngine\_Context\_Destroy等；***

***2）HIAI\_LLMEngine\_Executor\_GenerateAsync需要配合回调函数使用，尽量不要在后台还有任务的时候退出线程或进程，或提前卸载模型；***

***3）context的释放需要放在executor deint之后，否则在模型运行时释放context，则会crash；***

### 2.8.2 变量和结构体

| 结构体 | <span id="HIAI_LLMEngine_Executor">HIAI_LLMEngine_Executor</span> |
| :--- | :--- |
| 定义 |typedef struct HIAI_LLMEngine_Executor HIAI_LLMEngine_Executor;|
| 说明 | llm executor句柄 |
| 成员说明 | NA |

| 结构体 | <span id="HIAI_LLMEngine_ModelInfo">HIAI_LLMEngine_ModelInfo</span> |
| :--- | :--- |
| 定义 | typedef struct HIAI_LLMEngine_ModelInfo {<br>` `HIAI_LLMEngine_TokenizerType tokenizerType;<br>` `const char* tokenizerPath;<br>` `HIAI_LLMEngine_ModelType modelType;<br>` `const char* modelPath;<br>} HIAI_LLMEngine_ModelInfo; |
| 说明 | llm model info 句柄 ***（即将废弃，请勿使用）*** |
| 成员说明 | 略 |

| 枚举 | <span id="HIAI_LLMEngine_InferPerfMode">HIAI_LLMEngine_InferPerfMode</span> |
| :--- | :--- |
| 定义 | typedef enum {<br>` `HIAI_LLMEngine_InferPerf_UNSET = 0,<br>` `HIAI_LLMEngine_InferPerf_LOW,<br>` `HIAI_LLMEngine_InferPerf_MIDDLE,<br>` `HIAI_LLMEngine_InferPerf_HIGH,<br>` `HIAI_LLMEngine_InferPerf_EXTREME_HIGH,<br>} HIAI_LLMEngine_InferPerfMode; |
| 说明 | 推理性能等级 |
| 定义说明 | HIAI_LLMEngine_InferPerf_UNSET：默认值 |
| 定义说明 | HIAI_LLMEngine_InferPerf_LOW：低 |
| 定义说明 | HIAI_LLMEngine_InferPerf_MIDDLE：中 |
| 定义说明 | HIAI_LLMEngine_InferPerf_HIGH：高 |
| 定义说明 | HIAI_LLMEngine_InferPerf_EXTREME_HIGH：极致性能 |

### 2.8.3  句柄管理

| 方法原型 | HIAI\_LLMEngine\_Executor  \* HIAI\_LLMEngine\_Executor\_Create(void)  |
| :---- | :---- |
| 参数说明-入参 | NA |
| 参数说明-出参 | NA |
| 返回值类型 |  HIAI\_LLMEngine\_Executor  \* |
| 返回值说明 |  HIAI\_LLMEngine\_Executor  \*是一个对外部隐藏具体实现的结构体; |
| 说明 |  Executor句柄生成，返回一个句柄给用户 |

| 方法原型 | void  HIAI\_LLMEngine\_Executor\_Destroy(HIAI\_LLMEngine\_ Executor \*\* executor)   |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor  \*\* executor： Executor句柄的二级指针 |
| 参数说明-出参 | NA  |
| 返回值类型 |  void |
| 返回值说明 |  NA   |
| 说明 |  释放Executor句柄资源 |

### 2.8.4  功能类接口设计

#### 2.8.4.1 资源加卸载

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_Init( HIAI\_LLMEngine\_Executor\*  executor, const HIAI\_LLMEngine\_ModelInfo\*  modelInfo)  |
| :---- | :---- |
| 参数说明-入参 |  HIAI\_LLMEngine\_Executor\*  executor：句柄指针  |
| 参数说明-入参 |  const HIAI\_LLMEngine\_ModelInfo\*  modelInfo：模型参数信息；  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  ***注意：该接口即将被日落，请勿使用，请使用[HIAI_LLMEngine_InitOption](#HIAI_LLMEngine_InitOption)，并配合Executor\_Init\_Use\_Option接口来进行资源加载；***  |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_Init\_Use\_Option( HIAI\_LLMEngine\_Executor\*  executor, const  HIAI\_LLMEngine\_InitOption\* initOption)  |
| :--- | :--- |
| 参数说明-入参  |  HIAI\_LLMEngine\_Executor\*  executor：句柄指针 |
| 参数说明-入参  |  const  HIAI\_LLMEngine\_InitOption\* initOption：资源信息，[见HIAI_LLMEngine_InitOption说明](#HIAI_LLMEngine_InitOption) |
| 参数说明-出参  | NA  |
| 返回值类型  |  HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 使用initOption进行资源初始化 |

| 方法原型  |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_DeInit( HIAI\_LLMEngine\_Executor\*  executor)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：句柄指针  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 模型与词表资源卸载 |

#### 2.8.4.2  大模型加解密

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_UpdateWeight( HIAI\_LLMEngine\_Executor\*  executor, const char\*  weightName, size\_t offset, HIAI\_LM\_Buffer\*  weightBuffer)  |
| :---- | :---- |
| 参数说明-入参 |  HIAI\_LLMEngine\_Executor\*  executor：句柄指针 |
| 参数说明-入参 |  const char\*  weightName：权值文件名称 |
| 参数说明-入参 |  size\_t offset：权值文件大小 |
| 参数说明-入参 |  HIAI\_LM\_Buffer\*  weightBuffer：权重Buffer |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  更新大模型权值文件 |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_UpdateSpecWeight( HIAI\_LLMEngine\_Executor\*  executor, const char\*  weightName, size\_t offset, HIAI\_LM\_Buffer\*  weightBuffer)  |
| :--- | :--- |
| 参数说明-入参 |  HIAI\_LLMEngine\_Executor\*  executor：句柄指针 |
| 参数说明-入参 |  const char\*  weightName：权值文件名称 |
| 参数说明-入参 |  size\_t offset：权值文件大小 |
| 参数说明-入参 |  HIAI\_LM\_Buffer\*  weightBuffer：权重Buffer |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 更新投机推理小模型权值文件 |

#### 2.8.4.3 生成类接口

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_Generate( HIAI\_LLMEngine\_Executor\*  executor, HIAI\_LLMEngine\_Context\*  ctx, const char\* prompt)  |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：上下文句柄指针 |
| 参数说明-入参 | const char\* prompt：输入的文字prompt |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 执行模型推理 |

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_GenerateAsync( HIAI\_LLMEngine\_Executor\*  executor, HIAI\_LLMEngine\_Context\* ctx, const char\* prompt) |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\*  ctx：上下文句柄指针 |
| 参数说明-入参 | const char\* prompt：输入的文字prompt |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  执行模型异步推理，由Callback函数获取输出文本信息 |

#### 2.8.4.4 推理限速

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SetInferencePerfMode(HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_InferPerfMode inferPerfMode)      |
| :---- | :---- |
|  参数说明-入参  |  HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针；|
|  参数说明-入参  |  HIAI_LLMEngine_InferPerfMode inferPerfMode：推理性能模式枚举选项，[见HIAI_LLMEngine_InferPerfMode说明](#HIAI_LLMEngine_InferPerfMode)；|
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 设置对应模型的NPU推理性能模式 |

#### 2.8.4.5 多模态

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_InitVisionModel(HIAI_LLMEngine_Executor* executor, const HIAI_LMEngine_ModelInfo* modelInfo) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
| 参数说明-入参 | const HIAI_LMEngine_ModelInfo* modelInfo：视觉编码模型信息，[见HIAI_LMEngine_ModelInfo介绍](#HIAI_LMEngine_ModelInfo介绍) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 多模态视觉编码模型加载 |

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_DeInitVisionModel(HIAI_LLMEngine_Executor* executor) |
| :--- | :--- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针  |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 多模态视觉编码模型卸载 |

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MLLMGenerate(HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_Prompt* prompt) |
| :---- | :---- |
|  参数说明-入参  | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
|  参数说明-入参  | HIAI_LLMEngine_Context* ctx：上下文句柄指针 |
|  参数说明-入参  | HIAI_LLMEngine_Prompt* prompt：多模态的prompt句柄指针 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  多模态推理生成；<br>***注意：该接口也使用于单模态，单prompt句柄里面设置的只有text的时候，和文字生成的接口功能相同，后续归一到这个接口；*** |

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_MLLMGenerateAsync(HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_Prompt* prompt)  |
| :--- | :---- |
|  参数说明-入参  | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
|  参数说明-入参  | HIAI_LLMEngine_Context* ctx：上下文句柄指针 |
|  参数说明-入参  | HIAI_LLMEngine_Prompt* prompt：多模态的prompt句柄指针 |
| 参数说明-出参  | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  多模态异步推理生成；<br>***注意：该接口也适用于单模态，单prompt句柄里面设置的只有text的时候，和文字生成的接口功能相同，后续归一到这个接口；***  |

#### 2.8.4.6 LoRA

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateLora(HIAI_LLMEngine_Executor* executor, const char\* loraConfig)       |
| :---- | :---- |
| 参数说明-入参 |  HIAI\_LLMEngine\_Executor\*  executor Executor句柄；<br>const char* loraConfig lora配置文件路径； |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  大模型lora更新；<br>***注意：此处的loraconfig是在omg过程生成的，不要手动配置；*** |

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_UpdateSpecLora(HIAI_LLMEngine_Executor\* executor, const char\* loraConfig)  |
| :---- | :----- |
| 参数说明-入参  | HIAI\_LLMEngine\_Executor\*  executor Executor句柄；<br>const char* loraConfig 小模型lora配置文件路径； |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  投机推理小模型lora更新；<br>***注意：此处的loraconfig是在omg过程生成的，不要手动配置；*** |

#### 2.8.4.7 Prompt KV Cache

| 方法原型 |  HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_SavePromptKVCache(HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_PromptKVCache* promptKVCache)  |
| :---- | :---- |
|  参数说明-入参  | HIAI\_LLMEngine\_Executor* executor：executor句柄指针 |
|  参数说明-入参  | HIAI\_LLMEngine\_Context* ctx: 模型执行上下文句柄指针 |
|  参数说明-入参  | HIAI_LLMEngine_PromptKVCache* promptKVCache： PromptKVCache句柄 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  计算prefix prompt的kv cache并保存到PromptKVCache句柄中 |

| 方法原型 | HIAI_LLMEngine_Status HIAI_LLMEngine_Executor_LoadPromptKVCache(HIAI_LLMEngine_Executor* executor, HIAI_LLMEngine_Context* ctx, HIAI_LLMEngine_PromptKVCache* promptKVCache) |
| :--- | :--- |
|  参数说明-入参  | HIAI\_LLMEngine\_Executor* executor：executor句柄指针 |
|  参数说明-入参  | HIAI\_LLMEngine\_Context* ctx: 模型执行上下文句柄指针 |
|  参数说明-入参  | HIAI_LLMEngine_PromptKVCache* promptKVCache： PromptKVCache句柄 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 将PromptKVCache的cache加载到Tensor |

#### 2.8.4.8 暂停恢复

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_Suspend(HIAI\_LLMEngine\_Executor\*  executor)  |
| :--- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 |  暂停大模型推理；<br>***注意：<br>1. 在有模型运行的时候调用suspend会返回成功，如果没有模型运行调用会阻塞一段时间等待模型运行，如果没有则返回失败；<br>2. 调用suspend之后，释放模型资源请调用HIAI_LLMEngine_Executor_FreeResource、HIAI_LLMEngine_Executor_RestoreResource接口，不能直接调用Deinit接口；<br>3. suspend和resume需要匹配使用，在调用suspend之后必须要调用resume，才可以进行其他api调用；*** |

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_Resume( HIAI\_LLMEngine\_Executor\*  executor)  |
| :--- | :--- |
| 参数说明-入参  | HIAI\_LLMEngine\_Executor\*  executor：句柄指针 |
| 参数说明-出参  | NA  |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 恢复模型推理；<br>***注意：每个suspend至少匹配一个Resume；*** |

| 方法原型 | HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_FreeResource(HIAI\_LLMEngine\_Executor\*  executor, const HIAI_LLMEngine_ResourceOption* resourceOption)  |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
| 参数说明-入参 | const HIAI_LLMEngine_ResourceOption* resourceOption：resourceOption句柄指针，[见HIAI_LLMEngine_ResourceOption](#HIAI_LLMEngine_ResourceOption) |
| 参数说明-出参  |  NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 暂停后，释放模型资源；<br>***注意：这个接口是为了配合suspend使用的资源释放接口，不要在其他时候调用；*** |

| 方法原型 |  HIAI\_LLMEngine\_Status  HIAI\_LLMEngine\_Executor\_RestoreResource( HIAI\_LLMEngine\_Executor\*  executor, HIAI_LLMEngine_ResourceOption* resourceOption)  |
| :---- | :---- |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\*  executor：executor句柄指针 |
| 参数说明-入参 | const HIAI_LLMEngine_ResourceOption* resourceOption：resourceOption句柄指针，[见HIAI_LLMEngine_ResourceOption](#HIAI_LLMEngine_ResourceOption) |
| 参数说明-出参 | NA |
| 返回值类型 | HIAI\_LLMEngine\_Status  |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI_LLMEngine_Status) |
| 说明 | 暂停后，恢复模型资源；<br>***注意：这个接口是为了配合suspend使用的资源恢复接口，不要在其他时候调用；<br>在使用HIAI_LLMEngine_Executor_RestoreResource接口之后，HIAI_LLMEngine_Executor_Resume接口之前需要按照实际情况 进行lora的更新和解密权重的更新*** |

#### 2.8.4.9 端云协同首 token 加速

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Executor\_GetSentDataLen(HIAI\_LLMEngine\_Executor\* executor, HIAI\_LLMEngine\_Context\* ctx, uint32\_t\* len) |
| :------------ | :------------ |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\* executor: executor句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx: 模型执行上下文句柄指针 |
| 参数说明-入参 | uint32\_t\* len: 传输到云侧的上传json字符串长度的地址 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 获取传输到云侧的字符串长度 |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Executor\_GetSentData(HIAI\_LLMEngine\_Executor\* executor, HIAI\_LLMEngine\_Context\* ctx, char* sendData, uint32\_t len) |
| :------------ | :------------ |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\* executor: executor句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx: 模型执行上下文句柄指针 |
| 参数说明-入参 | char* sendData: 传输到云侧的上传json字符串地址 |
| 参数说明-入参 | uint32\_t len: 传输到云侧的上传json字符串长度 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 获取传输到云侧的字符串 |

| 方法原型 | HIAI\_LLMEngine\_Status HIAI\_LLMEngine\_Executor\_SetSparseParam(HIAI\_LLMEngine\_Executor\* executor, HIAI\_LLMEngine\_Context\* ctx, char* sparseParam) |
| :------------ | :------------ |
| 参数说明-入参 | HIAI\_LLMEngine\_Executor\* executor: executor句柄指针 |
| 参数说明-入参 | HIAI\_LLMEngine\_Context\* ctx: 模型执行上下文句柄指针 |
| 参数说明-入参 | char* sparseParam: 稀疏加速的参数，用下传json封装 |
| 返回值类型 | HIAI\_LLMEngine\_Status |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 设置稀疏加速的参数，即云侧下传的字符串。在GenerateAsync的prefill执行过程中设置有效，否则设置无效 |

#### 2.8.4.10 S2TT prefill阶段刷字

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefillTopk(const HIAI_LLMEngine_Context* ctx, uint32_t topk)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t topk`: topk的大小 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 设置prefill阶段需要的 topK |
| 约束和注意事项 | 设置的topk必须小于LLM输出的topk（通常设为16）。在调用这个接口的时候无法获得omc的topk，须在模型加载时判断，如不能满足，模型加载失败。 多轮对话场景，每轮都要重新设置才能正常生效|

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_SetPrefillCallbackTokenNum(const HIAI_LLMEngine_Context* ctx, uint32_t tokenNum)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t tokenNum`: tokenNum的大小 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 设置prefill阶段的 tokenNum，每推到 tokenNum 的整数倍调用回调函数，多轮对话场景，每轮都要重新设置才能正常生效|

| 方法原型 | `HIAI_LLMEngine_Status SetOnPrefillGenerateDoneFunc(HIAI_LLMEngine_Context* ctx, callbackFuncType func)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`callbackFuncType func`: 回调函数 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 设置prefill计算完成后的回调函数 |

| 方法原型 | `HIAI_LLMEngine_Status SetOnPrefillSomeTokenGenerateDoneFunc(HIAI_LLMEngine_Context* ctx, callbackFuncType func)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`callbackFuncType func`: 回调函数 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 设置prefill计算了部分topk之后的回调函数 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillOneTokenGeneration(HIAI_LLMEngine_Context* ctx, uint32_t len, uint32_t* indices, float* logits)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t len`: 用户传进来的indices的大小 = topk*chunk_len <br>`uint32_t* indices`: topk下标<br>`float* logits`: topk概率 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 获得Prefill部分token的topk的indices和logits |
| 约束和注意事项 | 校验len > topk*tokenNum，且注意是否有多线程冲突的情况。 |

| 方法原型 | `HIAI_LLMEngine_Status HIAI_LLMEngine_Context_GetPrefillAllTokenGeneration(HIAI_LLMEngine_Context* ctx, uint32_t len, uint32_t* indices, float* logits)` |
| :------------ | :------------ |
| 参数说明-入参 | `HIAI_LLMEngine_Context* ctx`: Context句柄<br>`uint32_t len`: 用户传进来的indices的大小 = topk*（已计算完成的topk数） <br>`uint32_t* indices`: topk下标<br>`float* logits`: topk概率 |
| 返回值类型 | `HIAI_LLMEngine_Status` |
| 返回值说明 | [见HIAI\_LLMEngine\_Status说明](#HIAI\_LLMEngine\_Status) |
| 说明 | 获得Prefill全部token的topk的indices和logits |
| 约束和注意事项 | 校验len > topk * (已计算完成的topk数)，推荐将 len 设成 topk * (所有token的num)，且注意是否有多线程冲突的情况。 |