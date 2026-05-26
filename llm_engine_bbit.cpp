#include <ctime>
#include <unistd.h>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <stdexcept>

#include "nlohmann/json.hpp"

#include "llm_engine_context.h"
#include "llm_engine_executor.h"
#include "llm_engine_kvcache.h"
#include "common/lm_engine_model_info.h"
#include "maintain/hiai_maint_wrapper.h"

#include "infra/base/assertion.h"
#include "framework/infra/log/log.h"
#include "infra/base/scope_guard.h"
#include "base/common/file_util/file_util.h"

#define LLM_ENGINE_BBIT_LOG_INFO(fmt, ...)              printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LLM_ENGINE_BBIT_LOG_ERROR(fmt, ...)             printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LLM_ENGINE_BBIT_LOG_WARNING(fmt, ...)           printf("[WARNING] " fmt "\n", ##__VA_ARGS__)

struct SpecTreeParam {
    HIAI_LLMEngine_SpecTreeType treeType = HIAI_LLMENGINE_STATIC_SPEC_TREE;
    uint32_t gamma = 0;
    float gammaThreshold = 0.0;
    std::vector<uint32_t> topHead;
    std::vector<std::vector<uint32_t>> topKSpec;
};

struct LLMEngineProcessParam {
    HIAI_LLMEngine_InferType inferType;
    HIAI_LLMEngine_TokenizerType tokenizerType;
    std::string tokenizerPath = "";
    HIAI_LLMEngine_ModelType modelType;
    std::string modelPath = "";
    std::string weightDir = "";
    std::string loraCfgPath = "";
    std::string baseLoraCfgPath = "";
    HIAI_LLMEngine_ModelType specModelType;
    std::string specModelPath = "";
    std::string specWeightDir = "";
    HIAI_LLMEngine_InferType compressInferType;
    HIAI_LLMEngine_TokenizerType compressTokenizerType;
    std::string compressTokenizerPath = "";
    HIAI_LLMEngine_ModelType compressModelType;
    std::string compressModelPath = "";
    std::string compressWeightDir = "";
    uint32_t pfxInitTokenLen = 6;
    std::string pmtCacheOperation = "";
    bool pmtCacheZeroCopy = false;
    std::string kvcacheHandle = "";
    std::string prefixPrompt = "";
    std::string specLoraCfgPath = "";
    std::string baseSpecLoraCfgPath = "";
    std::string prompt = "";
    HIAI_LLMEngine_SpecType specType;
    std::string draftText = "";
    uint32_t matchLen = 0;
    uint32_t windowSize = 0;
    std::vector<std::uint32_t> tokenids;
    std::vector<std::string> img;
    std::vector<std::string> imgEncoded;
    std::vector<std::string> audioEncoded;
    std::vector<std::string> audioEncodedSpec;
    HIAI_LLMEngine_ModelType vitModelType;
    std::string preprocessorConfigPath;
    std::string vitModelPath = "";
    std::string vitWeightDir = "";
    std::string mlpModelPath = "";
    std::string mlpWeightDir = "";
    uint32_t callbackFreq = 2;
    bool sampleFlag = false;
    uint32_t seed = 99;
    uint64_t topK = 100;
    float topP = 1.0;
    float temperature = 1.0;
    int64_t maxGenTokens = 64;
    std::vector<std::string> stopSeq;
    float repetitionPenalty = 1.0;
    uint32_t initTokenLen = 6;
    HIAI_LLMEngine_DynGammaStrategy gammaStrategy;
    uint32_t emaWindow = 1;
    std::vector<SpecTreeParam> treeList;
    float typicalScaling = 1.0;
    bool sampleGreedy = false;
    bool rejectScaling = false;
    float draftThreshold = 0.0;
    float compressRatio = 0.0;
    std::vector<std::string> forceTokens;
    std::string addtionalContext;
    std::string postPrompt;
    std::string prePosition;
    bool isAsync = true;
    int inferPerfMode = 0;
    bool multiTurnFlag = false;
    bool firstTokenAcclFlag = false;
    uint32_t setLatencyMs = 10;
    std::string firstTokenAcclMask = "";
    std::vector<std::string> firstTokenForceTokens;
    bool isSwitchKVCache = false;
    bool isS2TTEnable = false;
    uint32_t prefillTopk = 16;
    uint32_t prefillCallbackTokenNum = 10;
    uint32_t AllPrefillCallbackTokenNum = 200;
    std::string updateLoraCfgPath = "";
    std::string updateSpecLoraCfgPath = "";
    bool updateLoraFromBuffer = false;
    bool updateSpecLoraFromBuffer = false;
    bool updateWeight = true;
};

struct ImgPrompt {
    std::vector<void*> imgBufs;
    std::vector<size_t> imgBufsSizes;
    std::vector<void*> imgEncodedBufs;
    std::vector<size_t> imgEncodedBufsSizes;
    std::vector<HIAI_LMEngine_Image_Info> imgInfos;
};

struct ImgPromptInfo {
    std::vector<std::string> imgs;
    std::vector<std::string> imgEncodeds;
};

struct AudioPromptInfo {
    std::vector<std::string> audioEncodeds;
    std::vector<std::string> audioEncodedsSpec;
};

struct AudioPrompt {
    std::vector<void*> audioEncodedBufs;
    std::vector<size_t> audioEncodedBufsSizes;
};

double g_PrefillTime = 0.0;
double g_DecodeTime = 0.0;
bool g_LLMEngineGenerateAsyncFailed = false;
bool g_LLMEngineWaitFlag = false;
bool g_LLMEngineIsFirstToken = true;

std::string g_LLMEngineGeneration = "";
double g_LLMEngineTotalTimeMs = 0;
double g_LLMEngineInitTimeMs = 0;
double g_LLMEngineInitCompressTimeMs = 0;
double g_LLMEngineInitSpecTimeMs = 0;
double g_LLMEngineUpdateLoraTimeMs = 0;
double g_LLMEngineUpdateSpecLoraTimeMs = 0;
double g_LLMEngineDeInitTime = 0.0;
double g_LLMEnginePrefillTimeMs = 0;
double g_LLMEngineUpdateWeightTimeMs = 0;
double g_LLMEnginePrefillPerInferTimeMs = 0;
double g_LLMEngineDecodeTimeMs = 0;
double g_LLMEngineDecodePerTokenTimeMs = 0;
double g_LLMEngineDecodePerBlockTimeMs = 0;
uint64_t g_LLMEngineInputTokenCount = 0;
uint64_t g_LLMEngineOutputTokenCount = 0;
uint64_t g_LLMEngineCachedTokenCount = 0;
uint64_t g_LLMEngineDecodeNum = 0;

uint32_t g_testCount = 0;
double g_prefillTimePerToken = 0.0;
double g_decodeTimePerToken = 0.0;
double g_blockEfficiency = 0.0;
double g_decodeTimePerBlock = 0.0;

uint64_t g_LLMEnginePrefillLen = 32;

std::string g_historyPrompt = "";

std::string g_textPrompt = "";
std::vector<std::uint32_t> g_tokenidPrompt;

HIAI_LLMEngine_Context* g_LLMEngineContext = nullptr;
HIAI_LLMEngine_Executor* g_LLMEngineExecutor = nullptr;
HIAI_LLMEngine_InitOption* g_LLMEngineInitOption = nullptr;
HIAI_LMEngine_ModelInfo* g_LMEngineModelInfo = nullptr;
HIAI_LMEngine_ModelInfo* g_LMEngineVitModelInfo = nullptr;
HIAI_LMEngine_ModelInfo* g_LMEngineMlpModelInfo = nullptr;
HIAI_LMEngine_ModelInfo* g_LMEngineSpecModelInfo = nullptr;
HIAI_LLMEngine_InitOption* g_LLMEngineCompressInitOption = nullptr;
HIAI_LMEngine_ModelInfo* g_LMEngineCompressModelInfo = nullptr;
HIAI_LM_Buffer g_LMEngineModelBuffer;
HIAI_LM_Buffer g_LMEngineSpecModelBuffer;
HIAI_LM_Buffer g_LMEngineCompressModelBuffer;
HIAI_LM_Buffer g_LMEngineVitModelBuffer;
HIAI_LM_Buffer g_LMEngineMlpModelBuffer;
HIAI_LLMEngine_KVCache* g_KVCache1 = nullptr;
HIAI_LLMEngine_KVCache* g_KVCache2 = nullptr;
void* g_data = nullptr;
uint64_t g_size = 0;

bool g_mlp_release_flg = false;
bool g_vit_release_flg = false;

template<typename JType, typename VType>
VType GetJsonValue(JType& ctx, std::string k, VType defaultV)
{
    if (ctx.contains(k)) {
        return ctx[k];
    }
    return defaultV;
}

void InitTestInfo()
{
    g_testCount = 0;
    g_prefillTimePerToken = 0.0;
    g_decodeTimePerToken = 0.0;
    g_blockEfficiency = 0.0;
    g_decodeTimePerBlock = 0.0;
}

void UpdateTestInfo()
{
    g_testCount++;
    g_prefillTimePerToken += g_LLMEnginePrefillPerInferTimeMs;
    g_decodeTimePerToken += g_LLMEngineDecodePerTokenTimeMs;
    double block_efficiency = static_cast<double>(g_LLMEngineOutputTokenCount) /
        static_cast<double>(g_LLMEngineDecodeNum);
    g_blockEfficiency += block_efficiency;
    g_decodeTimePerBlock += g_LLMEngineDecodePerBlockTimeMs;
    LLM_ENGINE_BBIT_LOG_INFO("prefill_time_per_token, time = %lf", g_LLMEnginePrefillPerInferTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("decode_time_per_token, time = %lf", g_LLMEngineDecodePerTokenTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("block_efficiency = %lf", block_efficiency);
    LLM_ENGINE_BBIT_LOG_INFO("decode_time_per_block, time = %lf", g_LLMEngineDecodePerBlockTimeMs);
}

double GetTimeInUs()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return static_cast<double>(tv.tv_sec * 1000000 + tv.tv_usec);
}

void* LoadBinaryFileAll(std::string name, size_t& len)
{
    FILE *fp = fopen(name.c_str(), "rb");
    if (fp == nullptr) {
        FMK_LOGE("fopen %s failed.", name.c_str());
        return nullptr;
    }
    hiai::ScopeGuard fileGuard([&fp] () { fclose(fp); });

    if (fseek(fp, 0, SEEK_END) != 0) {
        FMK_LOGE("fseek end failed.");
        return nullptr;
    }
    len = ftell(fp);
    uint8_t* data = new uint8_t[len];
    if (data == nullptr) {
        FMK_LOGE("new data failed.");
        return nullptr;
    }

    if (fseek(fp, 0, 0) != 0) {
        FMK_LOGE("fseek begin failed.");
        return nullptr;
    }
    if (fread(data, sizeof(uint8_t), len, fp) != len) {
        FMK_LOGE("fread failed.");
        return nullptr;
    }
    FMK_LOGI("read file: %s, fd: %d, len: %lld, success.", name.c_str(), fp, len);
    return data;
}

std::vector<HIAI_LLMEngine_Prompt*> g_mllmPrompts;
std::vector<ImgPromptInfo> g_imgPromptInfos;
std::vector<ImgPrompt> g_imgPrompt;
std::vector<AudioPromptInfo> g_audioPromptInfos;
std::vector<AudioPrompt> g_audioPrompt;
bool LoadImgs(std::vector<std::string>& img, std::vector<void*>& imgBufs, std::vector<size_t>& imgBufsSizes)
{
    imgBufs.clear();
    imgBufsSizes.clear();
    for (auto& imgPath: img) {
        LLM_ENGINE_BBIT_LOG_INFO("load img: %s ...", imgPath.c_str());
        size_t len = 0;
        void* data = LoadBinaryFileAll(imgPath, len);
        if (data == nullptr) {
            LLM_ENGINE_BBIT_LOG_ERROR("load failed.");
            return false;
        }
        imgBufs.push_back(data);
        imgBufsSizes.push_back(len);
    }
    LLM_ENGINE_BBIT_LOG_INFO("load %lu img or img encoded, done.", imgBufs.size());
    for (size_t i = 0; i < imgBufs.size(); i++) {
        LLM_ENGINE_BBIT_LOG_INFO("-> data: %p, size: %lu.", imgBufs[i], imgBufsSizes[i]);
    }
    return true;
}

bool LLMEngineGenerateSync(HIAI_LLMEngine_Context* llmEngineContext,
    HIAI_LLMEngine_Executor* llmEngineExecutor, const std::string& llm_engine_prompt,
    std::vector<std::uint32_t>& tokenids,
    std::vector<std::string>& img, std::vector<std::string>& imgEncoded, size_t sentencesCtx_num)
{
    HIAI_LLMEngine_Status ret;
    std::uint32_t* tokenid = tokenids.data();
    if (!tokenids.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("input add tokenid.");
        // load tokenid
        if (HIAI_LLMEngine_Prompt_SetTokenIds(g_mllmPrompts[sentencesCtx_num],
            tokenid, tokenids.size()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetText failed.");
            return false;
        }
        ret = HIAI_LLMEngine_Executor_MLLMGenerate(llmEngineExecutor, llmEngineContext,
            g_mllmPrompts[sentencesCtx_num]);
    } else if (img.empty() && imgEncoded.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("input is only text.");
        ret = HIAI_LLMEngine_Executor_Generate(llmEngineExecutor, llmEngineContext,
            llm_engine_prompt.c_str());
    } else {
        LLM_ENGINE_BBIT_LOG_INFO("input add text.");
        // load text
        if (HIAI_LLMEngine_Prompt_SetText(g_mllmPrompts[sentencesCtx_num], llm_engine_prompt.c_str()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetText failed.");
            return false;
        }
        // run
        ret = HIAI_LLMEngine_Executor_MLLMGenerate(llmEngineExecutor, llmEngineContext, g_mllmPrompts[sentencesCtx_num]);
    }
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_Generate failed.");
        return false;
    }
    uint32_t len = 0;
    ret = HIAI_LLMEngine_Context_GetAllGenerationLen(llmEngineContext, &len);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGenerationLen failed.");
        return false;
    }

    char* generation = new char[len + 1];
    ret = HIAI_LLMEngine_Context_GetAllGeneration(llmEngineContext, generation, len);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGeneration failed.");
        return false;
    }
    generation[len] = '\0';
    LLM_ENGINE_BBIT_LOG_INFO("all generation len: %d.", len);
    LLM_ENGINE_BBIT_LOG_INFO("all generation: %s.", generation);

    delete[] generation;
    double totalTimeMs = 0;
        ret = HIAI_LLMEngine_Context_GetTotalTimeMs(llmEngineContext, &totalTimeMs);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetTotalTimeMs failed.");
            return true;
        }
        g_LLMEngineTotalTimeMs = totalTimeMs;

        double prefillTimeMs = 0;
        ret = HIAI_LLMEngine_Context_GetPrefillTimeMs(llmEngineContext, &prefillTimeMs);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetPrefillTimeMs failed.");
            return true;
        }
        g_LLMEnginePrefillTimeMs = prefillTimeMs;

        double decodeTimeMs = 0;
        ret = HIAI_LLMEngine_Context_GetDecodeTimeMs(llmEngineContext, &decodeTimeMs);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetDecodeTimeMs failed.");
            return true;
        }
        g_LLMEngineDecodeTimeMs = decodeTimeMs;

        uint64_t inputTokenCount = 0;
        ret = HIAI_LLMEngine_Context_GetInputTokenCount(llmEngineContext, &inputTokenCount);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetInputTokenCount failed.");
            return true;
        }
        g_LLMEngineInputTokenCount = inputTokenCount;

        uint64_t outputTokenCount = 0;
        ret = HIAI_LLMEngine_Context_GetOutputTokenCount(llmEngineContext, &outputTokenCount);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOutputTokenCount failed.");
            return true;
        }
        g_LLMEngineOutputTokenCount = outputTokenCount;

        uint64_t cachedTokenCount = 0;
        ret = HIAI_LLMEngine_Context_GetCachedTokenCount(llmEngineContext, &cachedTokenCount);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetCachedTokenCount failed.");
            return true;
        }
        g_LLMEngineCachedTokenCount = cachedTokenCount;

        uint64_t decodeNum = 0;
        ret = HIAI_LLMEngine_Context_GetDecodeNum(llmEngineContext, &decodeNum);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetDecodeNum failed.");
            return true;
        }
        g_LLMEngineDecodeNum = decodeNum;

        g_LLMEngineDecodePerTokenTimeMs = decodeTimeMs / outputTokenCount;
        g_LLMEngineDecodePerBlockTimeMs = decodeTimeMs / g_LLMEngineDecodeNum;

        LLM_ENGINE_BBIT_LOG_INFO("inputTokenCount: %lu.", inputTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("outputTokenCount: %lu.", outputTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("cachedTokenCount: %lu.", cachedTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("decodeNum: %lu.", decodeNum);
        LLM_ENGINE_BBIT_LOG_INFO("totalTimeMs: %.5f ms.", totalTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("prefillTimeMs: %.5f ms.", prefillTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("decodeTimeMs: %.5f ms.", decodeTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("decodeTimeMs per token: %.5f ms.", decodeTimeMs / outputTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("prefillLen: %lu.", g_LLMEnginePrefillLen);
        g_LLMEngineWaitFlag = false;
        g_LLMEnginePrefillPerInferTimeMs = prefillTimeMs / ceil(inputTokenCount / g_LLMEnginePrefillLen);
    return true;
}

bool LLMEngineGenerateAsync(HIAI_LLMEngine_Context* llmEngineContext,
    HIAI_LLMEngine_Executor* llmEngineExecutor, LLMEngineProcessParam& param, size_t sentencesCtx_num)
{
    std::string& llm_engine_prompt = param.prompt;
    std::vector<std::string>& img = param.img;
    std::vector<std::string>& imgEncoded = param.imgEncoded;
    std::vector<std::string>& audioEncoded = param.audioEncoded;
    std::vector<std::string>& audioEncodedSpec = param.audioEncodedSpec;
    std::uint32_t* tokenid = param.tokenids.data();

    if (param.isS2TTEnable) {
        auto ret = HIAI_LLMEngine_Context_SetPrefillTopk(llmEngineContext, param.prefillTopk);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetPrefillTopk failed.");
            return false;
        }

        ret = HIAI_LLMEngine_Context_SetPrefillCallbackTokenNum(llmEngineContext, param.prefillCallbackTokenNum);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetPrefillCallbackTokenNum failed.");
            return false;
        }
        
        void(*onPrefillGenerateDoneFunc)(const HIAI_LLMEngine_Context*) = [](const HIAI_LLMEngine_Context* ctx) {
            LLM_ENGINE_BBIT_LOG_INFO("onPrefillGenerateDoneFunc start");
            uint64_t len = 200 * 16;
            uint32_t *indices = new uint32_t[len + 1];
            float *logits = new float[len + 1];

            auto ret = HIAI_LLMEngine_Context_GetPrefillAllTokenGeneration(ctx, len, indices, logits);
            if (ret != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetPrefillAllTokenGeneration failed.");
                return;
            }

            indices[len] = '\0';
            logits[len] = '\0';
            fflush(stdout);
            for (uint32_t i = 0; i < len; i++) {
                if (i == 0) {
                    printf("[INFO] curr generation: %d ", indices[i]);
                } else {
                    printf("%d ", indices[i]);
                }
            }
            printf("\n");
            delete[] indices;
            delete[] logits;
        };
        ret = HIAI_LLMEngine_Context_SetOnPrefillGenerateDoneFunc(llmEngineContext,
            onPrefillGenerateDoneFunc);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetOnPrefillGenerateDoneFunc failed.");
            return false;
        }

        void(*onPrefillSomeTokenGenerateDoneFunc)(const HIAI_LLMEngine_Context*) = [](const HIAI_LLMEngine_Context* ctx) {
            uint32_t len = 10 * 16;
            uint32_t *indices = new uint32_t[len + 1];
            float *logits = new float[len + 1];

            auto ret = HIAI_LLMEngine_Context_GetPrefillOneTokenGeneration(ctx, len, indices, logits);
            if (ret != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetPrefillOneTokenGeneration failed.");
                return;
            }

            indices[len] = '\0';
            logits[len] = '\0';
            fflush(stdout);
            for (uint32_t i = 0; i < len; i++) {
                if (i == 0) {
                    printf("[INFO] curr generation: %d ", indices[i]);
                } else {
                    printf("%d ", indices[i]);
                }
            }
            printf("\n");
            delete[] indices;
            delete[] logits;
        };
        ret = HIAI_LLMEngine_Context_SetOnPrefillSomeTokenGenerateDoneFunc(llmEngineContext,
            onPrefillSomeTokenGenerateDoneFunc);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetOnPrefillSomeTokenGenerateDoneFunc failed.");
            return false;
        }
    }

    void(*onAllTokensGenerateDoneFunc)(const HIAI_LLMEngine_Context*) = [](const HIAI_LLMEngine_Context* ctx) {
        uint32_t len = 0;
        auto ret = HIAI_LLMEngine_Context_GetAllGenerationLen(ctx, &len);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGenerationLen failed.");
            return;
        }

        char* generation = new char[len + 1];
        ret = HIAI_LLMEngine_Context_GetAllGeneration(ctx, generation, len);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGeneration failed.");
            return;
        }
        generation[len] = '\0';
        printf("\n");
        fflush(stdout);
        LLM_ENGINE_BBIT_LOG_INFO("all generation len: %d.", len);
        LLM_ENGINE_BBIT_LOG_INFO("all generation: %s.", generation);

        g_LLMEngineGeneration = std::string(generation);
        delete[] generation;

        double totalTimeMs = 0;
        ret = HIAI_LLMEngine_Context_GetTotalTimeMs(ctx, &totalTimeMs);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetTotalTimeMs failed.");
            return;
        }
        g_LLMEngineTotalTimeMs = totalTimeMs;

        double prefillTimeMs = 0;
        ret = HIAI_LLMEngine_Context_GetPrefillTimeMs(ctx, &prefillTimeMs);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetPrefillTimeMs failed.");
            return;
        }
        g_LLMEnginePrefillTimeMs = prefillTimeMs;

        double decodeTimeMs = 0;
        ret = HIAI_LLMEngine_Context_GetDecodeTimeMs(ctx, &decodeTimeMs);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetDecodeTimeMs failed.");
            return;
        }
        g_LLMEngineDecodeTimeMs = decodeTimeMs;

        uint64_t inputTokenCount = 0;
        ret = HIAI_LLMEngine_Context_GetInputTokenCount(ctx, &inputTokenCount);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetInputTokenCount failed.");
            return;
        }
        g_LLMEngineInputTokenCount = inputTokenCount;

        uint64_t outputTokenCount = 0;
        ret = HIAI_LLMEngine_Context_GetOutputTokenCount(ctx, &outputTokenCount);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOutputTokenCount failed.");
            return;
        }
        g_LLMEngineOutputTokenCount = outputTokenCount;

        uint64_t cachedTokenCount = 0;
        ret = HIAI_LLMEngine_Context_GetCachedTokenCount(ctx, &cachedTokenCount);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetCachedTokenCount failed.");
            return;
        }
        g_LLMEngineCachedTokenCount = cachedTokenCount;

        uint64_t decodeNum = 0;
        ret = HIAI_LLMEngine_Context_GetDecodeNum(ctx, &decodeNum);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetDecodeNum failed.");
            return;
        }
        g_LLMEngineDecodeNum = decodeNum;

        g_LLMEngineDecodePerTokenTimeMs = decodeTimeMs / outputTokenCount;
        g_LLMEngineDecodePerBlockTimeMs = decodeTimeMs / g_LLMEngineDecodeNum;

        LLM_ENGINE_BBIT_LOG_INFO("inputTokenCount: %lu.", inputTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("outputTokenCount: %lu.", outputTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("cachedTokenCount: %lu.", cachedTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("decodeNum: %lu.", decodeNum);
        LLM_ENGINE_BBIT_LOG_INFO("totalTimeMs: %.5f ms.", totalTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("prefillTimeMs: %.5f ms.", prefillTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("decodeTimeMs: %.5f ms.", decodeTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("decodeTimeMs per token: %.5f ms.", decodeTimeMs / outputTokenCount);
        LLM_ENGINE_BBIT_LOG_INFO("prefillLen: %lu.", g_LLMEnginePrefillLen);
        g_LLMEngineWaitFlag = false;
        g_LLMEnginePrefillPerInferTimeMs = prefillTimeMs / ceil(inputTokenCount / g_LLMEnginePrefillLen);
    };

    auto ret = HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc(llmEngineContext,
        onAllTokensGenerateDoneFunc);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc failed.");
        return false;
    }

    void(*onSomeTokenGenerateDoneFunc)(const HIAI_LLMEngine_Context*) = [](const HIAI_LLMEngine_Context* ctx) {
        uint32_t len = 0;
        auto ret = HIAI_LLMEngine_Context_GetOneGenerationLen(ctx, &len);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOneGenerationLen failed.");
            return;
        }

        char* generation = new char[len + 1];
        ret = HIAI_LLMEngine_Context_GetOneGeneration(ctx, generation, len);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOneGeneration failed.");
            return;
        }
        generation[len] = '\0';
        if (g_LLMEngineIsFirstToken) {
            printf("[INFO] curr generation: %s", generation);
            g_LLMEngineIsFirstToken = false;
        } else {
            printf("%s", generation);
        }
        fflush(stdout);
        delete[] generation;
    };

    ret = HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc(llmEngineContext,
        onSomeTokenGenerateDoneFunc);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc failed.");
        return false;
    }

    void(*onGenerateAsyncFailedFunc)(const HIAI_LLMEngine_Context*) = [](const HIAI_LLMEngine_Context* ctx) {
        (void)ctx;
        g_LLMEngineWaitFlag = false;
        g_LLMEngineGenerateAsyncFailed = true;
    };

    ret = HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed(llmEngineContext,
        onGenerateAsyncFailedFunc);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetOnGenerateAsyncFailedFunc failed.");
        return false;
    }

    g_LLMEngineGenerateAsyncFailed = false;
    g_LLMEngineIsFirstToken = true;
    g_LLMEngineWaitFlag = true;
    if (!param.tokenids.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("input add tokenid.");
        // load tokenid
        if (HIAI_LLMEngine_Prompt_SetTokenIds(g_mllmPrompts[sentencesCtx_num],
            tokenid, param.tokenids.size()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetText failed.");
            return false;
        }
        ret = HIAI_LLMEngine_Executor_MLLMGenerate(llmEngineExecutor, llmEngineContext,
            g_mllmPrompts[sentencesCtx_num]);
    } else if (img.empty() && imgEncoded.empty() && audioEncoded.empty() && audioEncodedSpec.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("input is only text.");
        ret = HIAI_LLMEngine_Executor_GenerateAsync(llmEngineExecutor, llmEngineContext,
            llm_engine_prompt.c_str());
    } else {
        // load text
        if (HIAI_LLMEngine_Prompt_SetText(g_mllmPrompts[sentencesCtx_num], llm_engine_prompt.c_str()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetText failed.");
            return false;
        }
        if (!imgEncoded.empty() && img.empty()) {
            if (!LoadImgs(g_imgPromptInfos[sentencesCtx_num].imgEncodeds,
                g_imgPrompt[sentencesCtx_num].imgEncodedBufs,
                g_imgPrompt[sentencesCtx_num].imgEncodedBufsSizes)) {
                LLM_ENGINE_BBIT_LOG_ERROR("LoadImgs Encoded failed.");
                return false;
            }
            LLM_ENGINE_BBIT_LOG_INFO("input encoded add img.");
            if (HIAI_LLMEngine_Prompt_SetEncodedImageBufs(g_mllmPrompts[sentencesCtx_num],
                g_imgPrompt[sentencesCtx_num].imgEncodedBufs.data(),
                g_imgPrompt[sentencesCtx_num].imgEncodedBufsSizes.data(),
                g_imgPrompt[sentencesCtx_num].imgEncodedBufs.size()) != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetEncodedImageBufs failed.");
                return false;
            }
        }

        if (!audioEncoded.empty() || !audioEncodedSpec.empty()) {
            g_audioPromptInfos[sentencesCtx_num].audioEncodeds.insert(g_audioPromptInfos[sentencesCtx_num].audioEncodeds.end(),
                g_audioPromptInfos[sentencesCtx_num].audioEncodedsSpec.begin(),
                g_audioPromptInfos[sentencesCtx_num].audioEncodedsSpec.end());

            if (!LoadImgs(g_audioPromptInfos[sentencesCtx_num].audioEncodeds,
                g_audioPrompt[sentencesCtx_num].audioEncodedBufs,
                g_audioPrompt[sentencesCtx_num].audioEncodedBufsSizes)) {
                LLM_ENGINE_BBIT_LOG_ERROR("LoadAudio Encoded failed.");
                return false;
            }
        }

        if (!audioEncoded.empty() || !audioEncodedSpec.empty()) {
            LLM_ENGINE_BBIT_LOG_INFO("input encoded add audio.");
            if (HIAI_LLMEngine_Prompt_SetEncodedAudioBufs(g_mllmPrompts[sentencesCtx_num],
                g_audioPrompt[sentencesCtx_num].audioEncodedBufs.data(),
                g_audioPrompt[sentencesCtx_num].audioEncodedBufsSizes.data(),
                g_audioPrompt[sentencesCtx_num].audioEncodedBufs.size()) != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetEncodedImageBufs failed.");
                return false;
            }
        }
        // run
        ret = HIAI_LLMEngine_Executor_MLLMGenerateAsync(llmEngineExecutor, llmEngineContext, g_mllmPrompts[sentencesCtx_num]);
    }
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_GenerateAsync failed.");
        return false;
    }

    if (param.firstTokenAcclFlag) {
        uint32_t sentDatalen = 0;
        ret = HIAI_LLMEngine_Executor_GetSentDataLen(llmEngineExecutor, llmEngineContext, &sentDatalen);
        if (sentDatalen != 0) {
            char* sentData = new char[sentDatalen + 1];
            ret = HIAI_LLMEngine_Executor_GetSentData(llmEngineExecutor, llmEngineContext, sentData, sentDatalen);
            if (ret != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_GetSentData failed.");
                return false;
            }
            sentData[sentDatalen] = '\0';
            LLM_ENGINE_BBIT_LOG_INFO("sent data len: %d.", sentDatalen);
            LLM_ENGINE_BBIT_LOG_INFO("all sentdata: %s.", sentData);
            delete[] sentData;
        }
    }

    size_t loopCount = 0;
    while (g_LLMEngineWaitFlag) {
        loopCount++;
        if (param.firstTokenAcclFlag && loopCount == param.setLatencyMs) {
            ret = HIAI_LLMEngine_Executor_SetSparseParam(llmEngineExecutor, llmEngineContext, param.firstTokenAcclMask.c_str());
            if (ret != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_WARNING("HIAI_LLMEngine_Executor_SetSparseParam failed.");
            } else {
                LLM_ENGINE_BBIT_LOG_INFO("SetSparseParam SUCCESS!!!!!");
            }
        }
        usleep(1000);
    }
    if (g_LLMEngineGenerateAsyncFailed) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_GenerateAsync failed.");
        return false;
    }
    uint32_t len = 0;
    ret = HIAI_LLMEngine_Context_GetOneGenerationLen(llmEngineContext, &len);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOneGenerationLen failed.");
        return false;
    }

    char* generation = new char[len + 1];
    ret = HIAI_LLMEngine_Context_GetOneGeneration(llmEngineContext, generation, len);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOneGeneration failed.");
        return false;
    }
    generation[len] = '\0';
    LLM_ENGINE_BBIT_LOG_INFO("Check one generation len: %d, generation: %s", len, generation);
    delete[] generation;
    return true;
}

int InitModelBuffer(std::string& weightDir, std::string& modelPath, HIAI_LM_Buffer& modelBuffer, HIAI_LMEngine_ModelInfo* modelInfo)
{
    LLM_ENGINE_BBIT_LOG_INFO("InitModelBuffer::modelPath = %s", modelPath.c_str());
    if (modelInfo == nullptr) {
        LLM_ENGINE_BBIT_LOG_INFO("modelInfo == nullptr");
        return -1;
    }
    if (weightDir.size() > 0) {
        if (hiai::FileUtil::IsFileInImg(modelPath)) {
            LLM_ENGINE_BBIT_LOG_INFO("omc model is in img.");
            uint8_t* data = nullptr;
            size_t size = 0;
            if (hiai::FileUtil::LoadAsUint8(modelPath, size, data) != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("Load Img File failed.");
                return -1;
            }
            modelBuffer.size = size;
            modelBuffer.data = std::move(data);
            LLM_ENGINE_BBIT_LOG_INFO("modelBuffer size: %lu.", modelBuffer.size);
            if (modelBuffer.data == nullptr) {
                LLM_ENGINE_BBIT_LOG_ERROR("modelBuffer.data is nullptr.");
                return -1;
            }
            HIAI_LMEngine_ModelInfo_SetModelBuffer(modelInfo, &modelBuffer);
        } else {
            std::ifstream modelfileHandle(modelPath, std::ios::binary);
            if (!modelfileHandle.is_open()) {
                LLM_ENGINE_BBIT_LOG_ERROR("model path: %s, open failed.", modelPath.c_str());
                return -1;
            }
            modelfileHandle.seekg(0, std::ios::end);
            modelBuffer.size = modelfileHandle.tellg();
            modelBuffer.data = new uint8_t[modelBuffer.size];
            if (modelBuffer.data == nullptr) {
                LLM_ENGINE_BBIT_LOG_ERROR("model buffer data alloc failed.");
                return -1;
            }
            modelfileHandle.seekg(0, std::ios::beg);
            modelfileHandle.read(reinterpret_cast<char*>(modelBuffer.data),
                modelBuffer.size);
            LLM_ENGINE_BBIT_LOG_INFO("modelBuffer size: %lu.", modelBuffer.size);
            HIAI_LMEngine_ModelInfo_SetModelBuffer(modelInfo, &modelBuffer);
        }
    } else {
        LLM_ENGINE_BBIT_LOG_INFO("not use model buffer.");
    }
    return 0;
}

int UpdateWeight(LLMEngineProcessParam& param)
{
    if (!param.updateWeight) {
        return 0;
    }
    // update weight
    if (param.weightDir.size() > 0) {
        std::string weightPath = param.weightDir + "/SubGraph_0.weight";
        std::ifstream file(weightPath, std::ios::binary);
        if (file.is_open()) {
            size_t offset = 10000;
            size_t size = 1024 * 1024;
            file.seekg(offset);
            std::vector<char> buffer(size);
            file.read(buffer.data(), buffer.size());

            HIAI_LM_Buffer weightBuffer;
            weightBuffer.data = static_cast<void*>(buffer.data());
            weightBuffer.size = buffer.size();
            const char* weightName = "SubGraph_0";
            double updateWeightStart = GetTimeInUs();
            if (HIAI_LLMEngine_Executor_UpdateWeight(g_LLMEngineExecutor, weightName, offset, &weightBuffer) != HIAI_LLMEngine_SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("LLM Engine, HIAI_LLMEngine_Executor_UpdateWeight fail.");
                return -1;
            }
            g_LLMEngineUpdateWeightTimeMs = (GetTimeInUs() - updateWeightStart) / 1000.0;
            LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, update weight(1M) Done, takes %.2f ms", g_LLMEngineUpdateWeightTimeMs);
        }
    }
    return hiai::SUCCESS;
}


int InitVisionModel(LLMEngineProcessParam& param)
{
    if (param.vitModelPath.size() != 0) {
        LLM_ENGINE_BBIT_LOG_INFO("Init Vit Model...");
        LLM_ENGINE_BBIT_LOG_INFO("vit model path: %s", param.vitModelPath.c_str());
        LLM_ENGINE_BBIT_LOG_INFO("vit weight path: %s", param.vitWeightDir.c_str());
        LLM_ENGINE_BBIT_LOG_INFO("vit model type: %d", param.vitModelType);
        LLM_ENGINE_BBIT_LOG_INFO("vit preprocessor config path: %s", param.preprocessorConfigPath.c_str());
        g_LMEngineVitModelInfo = HIAI_LMEngine_ModelInfo_Create();
        HIAI_LMEngine_ModelInfo_SetModelPath(g_LMEngineVitModelInfo, param.vitModelPath.c_str());
        HIAI_LMEngine_ModelInfo_SetWeightDir(g_LMEngineVitModelInfo, param.vitWeightDir.c_str());
        HIAI_LMEngine_ModelInfo_SetModelType(g_LMEngineVitModelInfo, static_cast<size_t>(param.vitModelType));
        HIAI_LMEngine_ModelInfo_SetPreprocessorConfigPath(g_LMEngineVitModelInfo, param.preprocessorConfigPath.c_str());
        
        auto ret = InitModelBuffer(param.vitWeightDir, param.vitModelPath, g_LMEngineVitModelBuffer, g_LMEngineVitModelInfo);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("InitModelBuffer for g_LMEngineVitModelBuffer failed.");
            return -1;
        }

        g_vit_release_flg = true;

        if (HIAI_LLMEngine_Executor_InitVisionModel(g_LLMEngineExecutor,
            g_LMEngineVitModelInfo) != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_INFO("Init Vit Model, Failed.");
            return -1;
        }
        LLM_ENGINE_BBIT_LOG_INFO("Init Vit Model, Done.");
    }
    return hiai::SUCCESS;
}

int InitProjectionModel(LLMEngineProcessParam& param)
{
    if (param.mlpModelPath.size() != 0) {
        LLM_ENGINE_BBIT_LOG_INFO("Init Mlp Model...");
        LLM_ENGINE_BBIT_LOG_INFO("mlp model path: %s", param.mlpModelPath.c_str());
        LLM_ENGINE_BBIT_LOG_INFO("mlp weight path: %s", param.mlpWeightDir.c_str());
        g_LMEngineMlpModelInfo = HIAI_LMEngine_ModelInfo_Create();
        HIAI_LMEngine_ModelInfo_SetModelPath(g_LMEngineMlpModelInfo, param.mlpModelPath.c_str());
        HIAI_LMEngine_ModelInfo_SetWeightDir(g_LMEngineMlpModelInfo, param.mlpWeightDir.c_str());

        auto ret = InitModelBuffer(param.mlpWeightDir, param.mlpModelPath, g_LMEngineMlpModelBuffer, g_LMEngineMlpModelInfo);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("InitModelBuffer for g_LMEngineVitModelBuffer failed.");
            return -1;
        }

        g_mlp_release_flg = true;

        if (HIAI_LLMEngine_Executor_InitVisionProjection(g_LLMEngineExecutor,
            g_LMEngineMlpModelInfo) != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_INFO("Init Mlp Model, Failed.");
            return -1;
        }
        LLM_ENGINE_BBIT_LOG_INFO("Init Mlp Model, Done.");
    }
    return 0;
}

int InitVisionProcess(LLMEngineProcessParam& param)
{
    hiai::ScopeGuard vitModelBufferDataGuard([&] () {
        if (g_vit_release_flg) {
            delete[] reinterpret_cast<uint8_t*>(g_LMEngineVitModelBuffer.data);
            g_LMEngineVitModelBuffer.data = nullptr;
            g_LMEngineVitModelBuffer.size = 0;
        }
    });

    auto ret = InitVisionModel(param);
    if (ret != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("InitProjectionModel failed.");
        return -1;
    }

    hiai::ScopeGuard mlpModelBufferDataGuard([&] () {
        if (g_mlp_release_flg) {
            delete[] reinterpret_cast<uint8_t*>(g_LMEngineMlpModelBuffer.data);
            g_LMEngineMlpModelBuffer.data = nullptr;
            g_LMEngineMlpModelBuffer.size = 0;
        }
    });

    ret = InitProjectionModel(param);
    if (ret != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("InitProjectionModel failed.");
        return -1;
    }
    return 0;
}

int LLMEngineExecutorInit(LLMEngineProcessParam& param)
{
    (void)param;
    g_LLMEngineExecutor = HIAI_LLMEngine_Executor_Create();
    if (g_LLMEngineExecutor == nullptr) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_Create failed.");
        return -1;
    }
    return 0;
}

int LLMEnginePromptsInit(nlohmann::json& testCaseCtx)
{
    for (auto& sentencesCtx: testCaseCtx) {
        HIAI_LLMEngine_Prompt* mllmPrompt = HIAI_LLMEngine_Prompt_Create();
        if (mllmPrompt == nullptr) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_Create fail.");
            return -1;
        }
        g_mllmPrompts.push_back(mllmPrompt);
        ImgPromptInfo imgPromptInfo;
        imgPromptInfo.imgs = GetJsonValue(sentencesCtx, "img", std::vector<std::string>());
        imgPromptInfo.imgEncodeds = GetJsonValue(sentencesCtx, "imgEncoded", std::vector<std::string>());
        g_imgPromptInfos.push_back(imgPromptInfo);
        ImgPrompt imgPrompt;
        if (sentencesCtx.contains("imgInfo") && sentencesCtx["imgInfo"].is_array()) {
            for (size_t i = 0; i < sentencesCtx["imgInfo"].size(); ++i) {
                nlohmann::json imgInfoJson = sentencesCtx["imgInfo"][i];
                int64_t channel = GetJsonValue(imgInfoJson, "channel", int64_t(0));
                int64_t height = GetJsonValue(imgInfoJson, "height", int64_t(0));
                int64_t width = GetJsonValue(imgInfoJson, "width", int64_t(0));
                int64_t dataType = GetJsonValue(imgInfoJson, "dataType", int64_t(0));
                int64_t imageFormat = GetJsonValue(imgInfoJson, "imageFormat", int64_t(0));
                HIAI_LMEngine_Image_Info imgInfo = {
                    channel,
                    height,
                    width,
                    static_cast<HIAI_LMEngine_DataType>(dataType),
                    static_cast<HIAI_LMEngine_ImageFormat>(imageFormat)
                };
                imgPrompt.imgInfos.push_back(imgInfo);
            }
        }
        g_imgPrompt.push_back(imgPrompt);

        AudioPromptInfo audioPromtInfo;
        audioPromtInfo.audioEncodeds = GetJsonValue(sentencesCtx, "audioEncoded", std::vector<std::string>());
        audioPromtInfo.audioEncodedsSpec = GetJsonValue(sentencesCtx, "audioEncodedSpec", std::vector<std::string>());
        g_audioPromptInfos.push_back(audioPromtInfo);

        AudioPrompt audioPrompt;
        g_audioPrompt.push_back(audioPrompt);
    }
    return 0;
}

int LLMEngineInit(LLMEngineProcessParam& param)
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Init...");
    LLM_ENGINE_BBIT_LOG_INFO("inferType: %d", param.inferType);
    LLM_ENGINE_BBIT_LOG_INFO("model path: %s", param.modelPath.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("weight dir: %s", param.weightDir.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("spec model path: %s", param.specModelPath.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("spec weight dir: %s", param.specWeightDir.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("tokenizer path: %s", param.tokenizerPath.c_str());

    g_LLMEngineContext = HIAI_LLMEngine_Context_Create();
    if (g_LLMEngineContext == nullptr) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_Create failed.");
        return -1;
    }

    g_LLMEngineInitOption = HIAI_LLMEngine_InitOption_Create();
    if (g_LLMEngineInitOption == nullptr) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_Create failed.");
        return -1;
    }

    auto ret = HIAI_LLMEngine_InitOption_SetInferType(g_LLMEngineInitOption, param.inferType);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetInferType failed.");
        return -1;
    }

    ret = HIAI_LLMEngine_InitOption_SetTokenizer(g_LLMEngineInitOption, param.tokenizerType,
        param.tokenizerPath.c_str());
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetTokenizer failed.");
        return -1;
    }

    g_LMEngineModelInfo = HIAI_LMEngine_ModelInfo_Create();
    HIAI_LMEngine_ModelInfo_SetModelPath(g_LMEngineModelInfo, param.modelPath.c_str());
    HIAI_LMEngine_ModelInfo_SetWeightDir(g_LMEngineModelInfo, param.weightDir.c_str());

    ret = InitModelBuffer(param.weightDir, param.modelPath, g_LMEngineModelBuffer, g_LMEngineModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("InitModelBuffer for g_LMEngineModelBuffer failed.");
        return -1;
    }
    hiai::ScopeGuard modelBufferDataGuard([&] () {
        delete[] reinterpret_cast<uint8_t*>(g_LMEngineModelBuffer.data);
        g_LMEngineModelBuffer.data = nullptr;
        g_LMEngineModelBuffer.size = 0;
    });

    ret = HIAI_LLMEngine_InitOption_SetModel(g_LLMEngineInitOption, param.modelType, g_LMEngineModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetModel failed.");
        return -1;
    }

    g_LMEngineSpecModelInfo = HIAI_LMEngine_ModelInfo_Create();
    if (param.specWeightDir != "unknown dir." && param.specModelPath != "unknown path.") {
        HIAI_LMEngine_ModelInfo_SetModelPath(g_LMEngineSpecModelInfo, param.specModelPath.c_str());
        HIAI_LMEngine_ModelInfo_SetWeightDir(g_LMEngineSpecModelInfo, param.specWeightDir.c_str());
        ret = InitModelBuffer(param.specWeightDir, param.specModelPath, g_LMEngineSpecModelBuffer, g_LMEngineSpecModelInfo);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("InitModelBuffer for g_LMEngineSpecModelBuffer failed.");
            return -1;
        }
    }
    hiai::ScopeGuard specModelBufferDataGuard([&] () {
        delete[] reinterpret_cast<uint8_t*>(g_LMEngineSpecModelBuffer.data);
        g_LMEngineSpecModelBuffer.data = nullptr;
        g_LMEngineSpecModelBuffer.size = 0;
    });

    ret = HIAI_LLMEngine_InitOption_SetSpecModel(g_LLMEngineInitOption, param.specModelType, g_LMEngineSpecModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetSpecModel failed.");
        return -1;
    }

    if (param.isSwitchKVCache && HIAI_LLMEngine_InitOption_SetKVCacheSwitchableFlag(g_LLMEngineInitOption, param.isSwitchKVCache) != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetKVCacheSwitchableFlag failed.");
        return -1;
    }

    double initStart = GetTimeInUs();
    ret = HIAI_LLMEngine_Executor_Init_Use_Option(g_LLMEngineExecutor, g_LLMEngineInitOption);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_Init_Use_Option failed.");
        return -1;
    }
    g_LLMEngineInitTimeMs = (GetTimeInUs() - initStart) / 1000.0;
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Init_Use_Option takes %.2f ms", g_LLMEngineInitTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Init_Use_Option Done.");

    if (UpdateWeight(param) != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("UpdateWeight failed.");
    }

    return 0;
}

int LLMEngineInitCompress(LLMEngineProcessParam& param)
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Init Compress ...");
    LLM_ENGINE_BBIT_LOG_INFO("compressInferType: %d", param.compressInferType);
    LLM_ENGINE_BBIT_LOG_INFO("compress model path: %s", param.compressModelPath.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("compress weight dir: %s", param.compressWeightDir.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("compress tokenizer path: %s", param.compressTokenizerPath.c_str());

    LLM_ENGINE_BBIT_LOG_INFO("Unload SpecModel!!!");
    auto ret  = HIAI_LLMEngine_Executor_DeInitSpecModel(g_LLMEngineExecutor);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_DeInitSpecModel failed.");
        return -1;
    }
 
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, InitCompressModel start.");
    g_LLMEngineCompressInitOption = HIAI_LLMEngine_InitOption_Create();
    if (g_LLMEngineCompressInitOption == nullptr) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_Create g_LLMEngineCompressInitOption failed.");
        return -1;
    }

    ret = HIAI_LLMEngine_InitOption_SetInferType(g_LLMEngineCompressInitOption, param.compressInferType);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetInferType g_LLMEngineCompressInitOption failed.");
        return -1;
    }

    ret = HIAI_LLMEngine_InitOption_SetTokenizer(g_LLMEngineCompressInitOption, param.compressTokenizerType,
        param.compressTokenizerPath.c_str());
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetTokenizer g_LLMEngineCompressInitOption failed.");
        return -1;
    }

    g_LMEngineCompressModelInfo = HIAI_LMEngine_ModelInfo_Create();
    HIAI_LMEngine_ModelInfo_SetModelPath(g_LMEngineCompressModelInfo, param.compressModelPath.c_str());
    HIAI_LMEngine_ModelInfo_SetWeightDir(g_LMEngineCompressModelInfo, param.compressWeightDir.c_str());

    ret = InitModelBuffer(param.compressWeightDir, param.compressModelPath, g_LMEngineCompressModelBuffer, g_LMEngineCompressModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("InitModelBuffer for g_LMEngineCompressModelBuffer failed.");
        return -1;
    }
    hiai::ScopeGuard compressModelBufferDataGuard([&] () {
        delete[] reinterpret_cast<uint8_t*>(g_LMEngineCompressModelBuffer.data);
        g_LMEngineCompressModelBuffer.data = nullptr;
        g_LMEngineCompressModelBuffer.size = 0;
    });

    ret = HIAI_LLMEngine_InitOption_SetModel(g_LLMEngineCompressInitOption, param.compressModelType, g_LMEngineCompressModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetSpecModel g_LLMEngineCompressInitOption failed.");
        return -1;
    }

    double initCompressStart = GetTimeInUs();
    ret = HIAI_LLMEngine_Executor_InitCompressModel(g_LLMEngineExecutor, g_LLMEngineCompressInitOption);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_InitCompressModel failed.");
        return -1;
    }
    g_LLMEngineInitCompressTimeMs = (GetTimeInUs() - initCompressStart) / 1000.0;
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, InitCompressModel takes %.2f ms", g_LLMEngineInitCompressTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, InitCompressModel Done.");

    return 0;
}

HIAI_LM_Buffer* ReadFileToBuffer(const std::string& filePath, const std::string& logPrefix)
{
    std::ifstream fileHandle(filePath, std::ios::binary);
    if (!fileHandle.is_open()) {
        LLM_ENGINE_BBIT_LOG_ERROR("%smodel path: %s, open failed.", logPrefix.c_str(), filePath.c_str());
        return nullptr;
    }

    hiai::ScopeGuard fileGuard([&fileHandle]() { fileHandle.close(); });
    fileHandle.seekg(0, std::ios::end);
    uint64_t size = static_cast<uint64_t>(fileHandle.tellg());
    HIAI_LM_Buffer* buffer = new HIAI_LM_Buffer;
    buffer->size = size;
    buffer->data = new uint8_t[size];
    if (buffer->data == nullptr) {
        LLM_ENGINE_BBIT_LOG_ERROR("%smodel buffer data alloc failed.", logPrefix.c_str());
        delete buffer;
        return nullptr;
    }

    fileHandle.seekg(0, std::ios::beg);
    fileHandle.read(reinterpret_cast<char*>(buffer->data), size);
    LLM_ENGINE_BBIT_LOG_INFO("%sbuffer size: %lu.", logPrefix.c_str(), size);

    return buffer;
}

void ReleaseBuffer(HIAI_LM_Buffer* buffer)
{
    if (buffer != nullptr) {
        delete[] reinterpret_cast<uint8_t*>(buffer->data);
        buffer->data = nullptr;
        buffer->size = 0;
        delete buffer;
    }
}

int UpdateLoraFromBuffer(const std::string& modelPath, bool isSpecModel)
{
    std::string loraConfPath = "";
    if (hiai::FileUtil::RealpathUtil(modelPath, loraConfPath) != hiai::SUCCESS) {
        return -1;
    }

    HIAI_LM_Buffer* loraConfBuffer = ReadFileToBuffer(loraConfPath, "LLM Engine, ");
    if (loraConfBuffer == nullptr) {
        return -1;
    }

    hiai::ScopeGuard loraConfBufferGuard([loraConfBuffer]() { ReleaseBuffer(loraConfBuffer); });
    
    std::string::size_type len = (loraConfPath).find_last_of('.');
    if (len == std::string::npos) {
        FMK_LOGE("lora conf path error");
        return -1;
    }
    std::string dataPath = loraConfPath.substr(0, len) + ".loradata";
    std::string loraDataPath = "";
    if (hiai::FileUtil::RealpathUtil(dataPath, loraDataPath) != hiai::SUCCESS) {
        return -1;
    }
    HIAI_LM_Buffer* loraDataBuffer = ReadFileToBuffer(loraDataPath, "LLM Engine, ");
    if (loraDataBuffer == nullptr) {
        return -1;
    }
    hiai::ScopeGuard loraDataBufferGuard([loraDataBuffer]() { ReleaseBuffer(loraDataBuffer); });

    dataPath = loraConfPath.substr(0, len) + ".quantpara";
    std::string quantDataPath = "";
    if (hiai::FileUtil::RealpathUtil(dataPath, quantDataPath) != hiai::SUCCESS) {
        HIAI_LLMEngine_Status ret;
        if (isSpecModel) {
            ret = HIAI_LLMEngine_Executor_UpdateSpecLoraFromBuffer(
                g_LLMEngineExecutor, loraConfBuffer, loraDataBuffer, nullptr);
        } else {
            ret = HIAI_LLMEngine_Executor_UpdateLoraFromBuffer(
                g_LLMEngineExecutor, loraConfBuffer, loraDataBuffer, nullptr);
        }
        
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateLoraFromBuffer failed.");
            return -1;
        }
    } else {
        HIAI_LM_Buffer* loraQuatBuffer = ReadFileToBuffer(quantDataPath, "LLM Engine, ");
        if (loraQuatBuffer == nullptr) {
            return -1;
        }
        hiai::ScopeGuard loraQuatBufferGuard([loraQuatBuffer]() { ReleaseBuffer(loraQuatBuffer); });
        HIAI_LLMEngine_Status ret;
        if (isSpecModel) {
            ret = HIAI_LLMEngine_Executor_UpdateSpecLoraFromBuffer(
                g_LLMEngineExecutor, loraConfBuffer, loraDataBuffer, loraQuatBuffer);
        } else {
            ret = HIAI_LLMEngine_Executor_UpdateLoraFromBuffer(
                g_LLMEngineExecutor, loraConfBuffer, loraDataBuffer, loraQuatBuffer);
        }
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateLoraFromBuffer failed.");
            return -1;
        }
    }
    return 0;
}

int LLMEngineUpdateLora(LLMEngineProcessParam& param)
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora...");
    LLM_ENGINE_BBIT_LOG_INFO("lora config file path: %s", param.loraCfgPath.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("spec lora config file path: %s", param.specLoraCfgPath.c_str());

    double updateLoraStart = GetTimeInUs();
    if (!param.loraCfgPath.empty()) {
        if (!param.updateLoraFromBuffer) {
            auto ret = HIAI_LLMEngine_Executor_UpdateLora(g_LLMEngineExecutor, param.loraCfgPath.c_str());
            if (ret != HIAI_LLMEngine_SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateLora failed.");
                return -1;
            }
        } else {
            if (UpdateLoraFromBuffer(param.loraCfgPath, false) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("UpdateLoraFromBuffer failed.");
                return -1;
            }
        }
        g_LLMEngineUpdateLoraTimeMs = (GetTimeInUs() - updateLoraStart) / 1000.0;
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora takes %.2f ms", g_LLMEngineUpdateLoraTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora Done.");
    }
    if (!param.baseLoraCfgPath.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora, switch to baseModel");
        auto ret = HIAI_LLMEngine_Executor_UpdateLora(g_LLMEngineExecutor, param.baseLoraCfgPath.c_str());
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateLora switch to baseModel failed.");
            return -1;
        }
    }
    double updateSpecLoraStart = GetTimeInUs();
    if (!param.specLoraCfgPath.empty()) {
        if (!param.updateSpecLoraFromBuffer) {
            auto ret = HIAI_LLMEngine_Executor_UpdateSpecLora(g_LLMEngineExecutor, param.specLoraCfgPath.c_str());
            if (ret != HIAI_LLMEngine_SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateSpecLora failed.");
                return -1;
            }
        } else {
            if (UpdateLoraFromBuffer(param.specLoraCfgPath, true) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("UpdateLoraFromBuffer failed.");
                return -1;
            }
        }

        g_LLMEngineUpdateSpecLoraTimeMs = (GetTimeInUs() - updateSpecLoraStart) / 1000.0;
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora takes %.2f ms", g_LLMEngineUpdateSpecLoraTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora Done.");
    }
    if (!param.baseSpecLoraCfgPath.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora, switch to baseModel");
        auto ret = HIAI_LLMEngine_Executor_UpdateSpecLora(g_LLMEngineExecutor, param.baseSpecLoraCfgPath.c_str());
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateSpecLora switch to baseModel failed.");
            return -1;
        }
    }
    return 0;
}

void LLMEngineVisionDeInit()
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine Vision, DeInit...");
    if (g_LLMEngineExecutor != nullptr && g_LMEngineVitModelInfo != nullptr) {
        HIAI_LLMEngine_Executor_DeInitVisionModel(g_LLMEngineExecutor);
    }
    if (g_LLMEngineExecutor != nullptr && g_LMEngineMlpModelInfo != nullptr) {
        HIAI_LLMEngine_Executor_DeInitVisionProjection(g_LLMEngineExecutor);
    }
}

void LLMEngineDeInit()
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, DeInit...");
    double deinitStart = GetTimeInUs();
    if (g_LLMEngineExecutor != nullptr && g_LLMEngineCompressInitOption != nullptr) {
        HIAI_LLMEngine_Executor_DeInitCompressModel(g_LLMEngineExecutor);
    }
    HIAI_LLMEngine_Executor_Deinit(g_LLMEngineExecutor);
    HIAI_LLMEngine_Executor_Destroy(&g_LLMEngineExecutor);
    HIAI_LLMEngine_Context_Destroy(&g_LLMEngineContext);
    HIAI_LMEngine_ModelInfo_Destroy(&g_LMEngineModelInfo);
    HIAI_LMEngine_ModelInfo_Destroy(&g_LMEngineSpecModelInfo);
    HIAI_LLMEngine_InitOption_Destroy(&g_LLMEngineInitOption);
    HIAI_LMEngine_ModelInfo_Destroy(&g_LMEngineCompressModelInfo);
    HIAI_LLMEngine_InitOption_Destroy(&g_LLMEngineCompressInitOption);
    HIAI_LLMEngine_Prompt_Destroy(&g_mllmPrompts[0]);
    if (g_KVCache1 != nullptr) {
        HIAI_LLMEngine_DestroyKVCache(&g_KVCache1);
    }
    if (g_KVCache2 != nullptr) {
        HIAI_LLMEngine_DestroyKVCache(&g_KVCache2);
    }
    g_mllmPrompts.clear();
    g_LLMEngineDeInitTime = (GetTimeInUs() - deinitStart) / 1000.0;
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, DeInit Done.");
}

int LLMEngineDeInitCompress(LLMEngineProcessParam& param)
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, DeInit Compress...");
    auto ret = HIAI_LLMEngine_Executor_DeInitCompressModel(g_LLMEngineExecutor);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_DeInitCompressModel failed.");
    }

    if (g_LMEngineSpecModelInfo != nullptr)  {
        HIAI_LMEngine_ModelInfo_Destroy(&g_LMEngineSpecModelInfo);
    }
    g_LMEngineSpecModelInfo = HIAI_LMEngine_ModelInfo_Create();
    HIAI_LMEngine_ModelInfo_SetModelPath(g_LMEngineSpecModelInfo, param.specModelPath.c_str());
    HIAI_LMEngine_ModelInfo_SetWeightDir(g_LMEngineSpecModelInfo, param.specWeightDir.c_str());

    ret = InitModelBuffer(param.specWeightDir, param.specModelPath, g_LMEngineSpecModelBuffer, g_LMEngineSpecModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("InitModelBuffer for g_LMEngineSpecModelBuffer failed.");
        return -1;
    }

    hiai::ScopeGuard specModelBufferDataGuard([&] () {
        delete[] reinterpret_cast<uint8_t*>(g_LMEngineSpecModelBuffer.data);
        g_LMEngineSpecModelBuffer.data = nullptr;
        g_LMEngineSpecModelBuffer.size = 0;
    });

    ret = HIAI_LLMEngine_InitOption_SetSpecModel(g_LLMEngineInitOption, param.specModelType, g_LMEngineSpecModelInfo);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_InitOption_SetSpecModel failed.");
        return -1;
    }

    double initSpecStart = GetTimeInUs();
    ret = HIAI_LLMEngine_Executor_InitSpecModel(g_LLMEngineExecutor, g_LLMEngineInitOption);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_InitSpecModel failed.");
        return -1;
    }
    g_LLMEngineInitSpecTimeMs = (GetTimeInUs() - initSpecStart) / 1000.0;
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, InitSpecModel takes %.2f ms", g_LLMEngineInitSpecTimeMs);

    double updateSpecLoraStart = GetTimeInUs();
    if (!param.specLoraCfgPath.empty()) {
        auto ret = HIAI_LLMEngine_Executor_UpdateSpecLora(g_LLMEngineExecutor, param.specLoraCfgPath.c_str());
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateSpecLora failed.");
            return -1;
        }
        g_LLMEngineUpdateSpecLoraTimeMs = (GetTimeInUs() - updateSpecLoraStart) / 1000.0;
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora takes %.2f ms", g_LLMEngineUpdateSpecLoraTimeMs);
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora Done.");
    }

    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, DeInitCompressModel Done.");
    return 0;
}

void ClearCache()
{
    system("echo 1 > /proc/sys/vm/drop_caches");
    system("echo 2 > /proc/sys/vm/drop_caches");
    system("echo 3 > /proc/sys/vm/drop_caches");
    system("sync");
}

void DumpBinaryToFile(std::string name, void* dataVoid, size_t len)
{
    uint8_t* data = reinterpret_cast<uint8_t*>(dataVoid);

    std::string filePath = name;
    LLM_ENGINE_BBIT_LOG_INFO("DumpBinaryToFile: %s", filePath.c_str());

    FILE *fp = fopen(filePath.c_str(), "wb");
    if (fp == nullptr) {
        FMK_LOGE("%s open failed !!! ", filePath.c_str());
        return;
    }

    size_t write_size = fwrite(data, 1, len, fp);
    if (write_size != len) {
        fclose(fp);
        FMK_LOGE("write op output file failed !!!");
        return;
    }
    fclose(fp);
}

bool LoadBinaryFile(std::string name, void* dataVoid, size_t len)
{
    uint8_t* data = reinterpret_cast<uint8_t*>(dataVoid);
    FILE *fp = fopen(name.c_str(), "rb");
    if (fp == nullptr) {
        FMK_LOGI("LoadBinaryFile Failed: Open File %s failed !!! ", name.c_str());
        return false;
    }
    size_t read_size = fread(data, 1, len, fp);
    if (read_size != len) {
        fclose(fp);
        LLM_ENGINE_BBIT_LOG_INFO("LoadBinaryFile Failed: Read %s content failed !!! ", name.c_str());
        return false;
    }
    fclose(fp);
    return true;
}

int LLMEngineRun(LLMEngineProcessParam& param, size_t sentencesCtx_num)
{
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Run...");

    LLM_ENGINE_BBIT_LOG_INFO("prefixPrompt: %s", param.prefixPrompt.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("prompt: %s", param.prompt.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("draftText: %s", param.draftText.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("matchLen: %d", param.matchLen);
    LLM_ENGINE_BBIT_LOG_INFO("windowSize: %d", param.windowSize);
    LLM_ENGINE_BBIT_LOG_INFO("specType: %d", param.specType);
    LLM_ENGINE_BBIT_LOG_INFO("max gen tokens: %ld", param.maxGenTokens);
    LLM_ENGINE_BBIT_LOG_INFO("cb freq: %d", param.callbackFreq);
    LLM_ENGINE_BBIT_LOG_INFO("seed: %d", param.seed);
    LLM_ENGINE_BBIT_LOG_INFO("topk: %lu", param.topK);
    LLM_ENGINE_BBIT_LOG_INFO("topp: %f", param.topP);
    LLM_ENGINE_BBIT_LOG_INFO("temperature: %f", param.temperature);
    LLM_ENGINE_BBIT_LOG_INFO("async mode: %d", param.isAsync);
    LLM_ENGINE_BBIT_LOG_INFO("do sample: %d", param.sampleFlag);
    LLM_ENGINE_BBIT_LOG_INFO("repetitionPenalty: %f", param.repetitionPenalty);
    LLM_ENGINE_BBIT_LOG_INFO("initTokenLen: %d", param.initTokenLen);
    for (std::string stopWord : param.stopSeq) {
        LLM_ENGINE_BBIT_LOG_INFO("stopWord: %s", stopWord.c_str());
    }
    LLM_ENGINE_BBIT_LOG_INFO("inferPerfMode: %d", param.inferPerfMode);

    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetCallbackFreq(g_LLMEngineContext, param.callbackFreq) ==
        hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetSeed(g_LLMEngineContext, param.seed) == hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetTopK(g_LLMEngineContext, param.topK) == hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetTopP(g_LLMEngineContext, param.topP) == hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetTemperature(g_LLMEngineContext, param.temperature) ==
        hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetMaxGenTokens(g_LLMEngineContext, param.maxGenTokens) ==
        hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetDoSampleFlag(g_LLMEngineContext, param.sampleFlag) ==
        hiai::SUCCESS, -1);
    char* stopSeq[param.stopSeq.size()];
    for (size_t i = 0; i < param.stopSeq.size(); i++) {
        stopSeq[i] = param.stopSeq[i].data();
    }
    HIAI_LLMEngine_Context_SetStopSeq(g_LLMEngineContext, stopSeq, param.stopSeq.size());
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetRepetitionPenalty(g_LLMEngineContext, param.repetitionPenalty) ==
        hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetInitTokenLen(g_LLMEngineContext, param.initTokenLen) ==
        hiai::SUCCESS, -1);
    if (param.inferType == HIAI_LLMENGINE_TREESPEC_INFERTYPE && param.specType == HIAI_LLMENGINE_TREE_SPECTYPE) {
        LLM_ENGINE_BBIT_LOG_INFO("gammaStrategy: %d", param.gammaStrategy);
        LLM_ENGINE_BBIT_LOG_INFO("emaWindow: %d", param.emaWindow);
        LLM_ENGINE_BBIT_LOG_INFO("typicalScaling: %f", param.typicalScaling);
        LLM_ENGINE_BBIT_LOG_INFO("draftThreshold: %f", param.draftThreshold);
        LLM_ENGINE_BBIT_LOG_INFO("sampleGreedy: %d", param.sampleGreedy);
        LLM_ENGINE_BBIT_LOG_INFO("rejectScaling: %d", param.rejectScaling);
        HIAI_LLMEngine_Context_ClearSpecTree(g_LLMEngineContext);
        for (uint32_t t = 0; t < param.treeList.size(); t++) {
            LLM_ENGINE_BBIT_LOG_INFO("tree%d gamma : %d", t, param.treeList[t].gamma);
            LLM_ENGINE_BBIT_LOG_INFO("tree%d treeType : %d", t, param.treeList[t].treeType);
            LLM_ENGINE_BBIT_LOG_INFO("tree%d gammaThreshold : %f", t, param.treeList[t].gammaThreshold);
            uint32_t topHead[param.treeList[t].gamma];
            uint32_t topKSpec[param.treeList[t].gamma][16];
            for (size_t i = 0; i < param.treeList[t].gamma; i++) {
                topHead[i] = param.treeList[t].topHead[i];
                for (size_t j = 0; j < param.treeList[t].topKSpec[i].size(); j++) {
                    topKSpec[i][j] = param.treeList[t].topKSpec[i][j];
                }
            }
            HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetSpecTreeType(g_LLMEngineContext, t, param.treeList[t].treeType) == hiai::SUCCESS, -1);
            HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetSpecTreeShape(g_LLMEngineContext, t, param.treeList[t].gamma, topHead,
                (const uint32_t**) topKSpec, 16) == hiai::SUCCESS, -1);
            HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetSpecTreeConfidenceThresh(g_LLMEngineContext, t, param.treeList[t].gammaThreshold) ==
                hiai::SUCCESS, -1);
        }
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetDynGammaStrategy(g_LLMEngineContext, param.gammaStrategy) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetDynGammaEmaWindow(g_LLMEngineContext, param.emaWindow) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetTypicalScaling(g_LLMEngineContext, param.typicalScaling) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetDraftThreshold(g_LLMEngineContext, param.draftThreshold) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetSampleGreedy(g_LLMEngineContext, param.sampleGreedy) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetRejectScaling(g_LLMEngineContext, param.rejectScaling) == hiai::SUCCESS, -1);
    } else if (param.inferType == HIAI_LLMENGINE_TREESPEC_INFERTYPE && param.specType == HIAI_LLMENGINE_TEXT_SPECTYPE) {
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetSpecType(g_LLMEngineContext, param.specType) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetDraftText(g_LLMEngineContext, param.draftText.c_str()) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetMatchLen(g_LLMEngineContext, param.matchLen) == hiai::SUCCESS, -1);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetWindowSize(g_LLMEngineContext, param.windowSize) == hiai::SUCCESS, -1);
    }
    if (param.firstTokenAcclFlag) {
        char* forceTokens[param.forceTokens.size()];
        for (size_t i = 0; i < param.forceTokens.size(); i++) {
            forceTokens[i] = param.forceTokens[i].data();
        }
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetForceTokens(g_LLMEngineContext, forceTokens, param.forceTokens.size()) ==
            hiai::SUCCESS, -1);
    }
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Executor_SetInferencePerfMode(g_LLMEngineExecutor,
        static_cast<HIAI_LLMEngine_InferPerfMode>(param.inferPerfMode)) == hiai::SUCCESS, -1);

    if (!(param.isAsync)) {
        LLM_ENGINE_BBIT_LOG_INFO("run on sync mode.");
        if (!LLMEngineGenerateSync(g_LLMEngineContext,
            g_LLMEngineExecutor, param.prompt, param.tokenids, param.img, param.imgEncoded, sentencesCtx_num)) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineGenerateSync failed.");
            return -1;
        }
    } else {
        LLM_ENGINE_BBIT_LOG_INFO("run on async mode.");
        if (!LLMEngineGenerateAsync(g_LLMEngineContext, g_LLMEngineExecutor, param, sentencesCtx_num)) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineGenerateAsync failed.");
            return -1;
        }
    }
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Run Done.");
    return 0;
}

int LLMEngineCompress(LLMEngineProcessParam& param)
{
    LLM_ENGINE_BBIT_LOG_INFO("Run Prompt Compress!!!");
    std::string promptForShow;
    int32_t periodPos = param.prompt.find("\n");
    if (periodPos != std::string::npos) {
        promptForShow = param.prompt.substr(0, periodPos) + "...";
    } else {
        promptForShow = param.prompt.substr(0, 100) + "...";
    }
    LLM_ENGINE_BBIT_LOG_INFO("prompt: %s", promptForShow.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("addtionalContext: %s", param.addtionalContext.c_str());
    LLM_ENGINE_BBIT_LOG_INFO("postPrompt: %s", param.postPrompt.c_str());

    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetCompressRatio(g_LLMEngineContext, param.compressRatio) ==
        hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetAdditionalContext(g_LLMEngineContext, param.addtionalContext.c_str()) ==
        hiai::SUCCESS, -1);
    char* forceTokens[param.forceTokens.size()];
    for (size_t i = 0; i < param.forceTokens.size(); i++) {
        forceTokens[i] = param.forceTokens[i].data();
    }
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetForceTokens(g_LLMEngineContext, forceTokens, param.forceTokens.size()) ==
        hiai::SUCCESS, -1);
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Executor_CompressPrompt(g_LLMEngineExecutor, g_LLMEngineContext, param.prompt.c_str()) ==
        hiai::SUCCESS, -1);
    uint32_t compressedPromptLen = 0;
    HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_GetCompressPromptLen(g_LLMEngineContext, &compressedPromptLen) ==
        hiai::SUCCESS, -1);
    char* generation = new char[compressedPromptLen + 1];
    HIAI_LLMEngine_Status ret = HIAI_LLMEngine_Context_GetCompressPrompt(g_LLMEngineContext, generation, compressedPromptLen);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetCompressPromptLen failed.");
    }
    generation[compressedPromptLen] = '\0';
    LLM_ENGINE_BBIT_LOG_INFO("compress len: %d.", compressedPromptLen);
    LLM_ENGINE_BBIT_LOG_INFO("compress result: %s.", generation);

    param.prompt = param.prePosition + generation + param.postPrompt;
    
    delete[] generation;
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Compress Done.");
    return 0;
}

uint64_t GetMaxDDRFreq()
{
    static const std::string path = "/sys/class/devfreq/ddrfreq/available_frequencies";
    std::ifstream ifs(path, std::ios::in);
    if (!ifs) {
        LLM_ENGINE_BBIT_LOG_WARNING("file: %s open failed.", path.c_str());
        return 0;
    }
    std::string ctx;
    getline(ifs, ctx);
    std::stringstream ss;
    ss << ctx;
    uint64_t maxDDRFreqMhz = 0;
    while (!ss.eof()) {
        uint64_t freq;
        ss >> freq;
        freq /= 1000000;
        maxDDRFreqMhz = freq > maxDDRFreqMhz ? freq : maxDDRFreqMhz;
    }
    return maxDDRFreqMhz;
}

void PrintTimeInfo()
{
    uint64_t maxDDRFreqMhz = GetMaxDDRFreq();
    printf("PERF_PROJECT : module = FMK, currentFunc = rdv_case_LLM_Engine_ddrMax%luMHz, "
        "callerFunc = init_time, status = 2, time = %lf \n", maxDDRFreqMhz, g_LLMEngineInitTimeMs);
    printf("PERF_PROJECT : module = FMK, currentFunc = rdv_case_LLM_Engine_ddrMax%luMHz, "
        "callerFunc = deinit_time, status = 2, time = %lf \n", maxDDRFreqMhz, g_LLMEngineDeInitTime);
    if (g_testCount != 0) {
        printf("PERF_PROJECT : module = FMK, currentFunc = rdv_case_LLM_Engine_ddrMax%luMHz, "
            "callerFunc = prefill_time_per_token, status = 2, time = %lf \n",
            maxDDRFreqMhz, g_prefillTimePerToken / g_testCount);
        printf("PERF_PROJECT : module = FMK, currentFunc = rdv_case_LLM_Engine_ddrMax%luMHz, "
            "callerFunc = decode_time_per_token, status = 2, time = %lf \n",
            maxDDRFreqMhz, g_decodeTimePerToken / g_testCount);
        printf("PERF_PROJECT : module = FMK, currentFunc = rdv_case_LLM_Engine_ddrMax%luMHz, "
            "callerFunc = block_efficiency, status = 2, block_efficiency = %lf \n",
            maxDDRFreqMhz, g_blockEfficiency / g_testCount);
        printf("PERF_PROJECT : module = FMK, currentFunc = rdv_case_LLM_Engine_ddrMax%luMHz, "
            "callerFunc = decode_time_per_block, status = 2, time = %lf \n",
            maxDDRFreqMhz, g_decodeTimePerBlock / g_testCount);
    }
}

template<typename CtxType>
void TreeSpecInferParam(LLMEngineProcessParam& param, CtxType& sentencesCtx)
{
    if (param.inferType != HIAI_LLMENGINE_TREESPEC_INFERTYPE) {
        return;
    }
    param.treeList.clear();
    if (!sentencesCtx.contains("specTree")) {
        SpecTreeParam tree;
        tree.topHead.clear();
        tree.topKSpec.clear();
        tree.gamma = GetJsonValue(sentencesCtx, "gamma", 3);
        if (sentencesCtx.contains("topHead")) {
            auto topHeadSize = sentencesCtx["topHead"].size();
            for (size_t i = 0; i < topHeadSize; i++) {
                tree.topHead.push_back(sentencesCtx["topHead"][i]);
            }
        }
        if (sentencesCtx.contains("topKSpec")) {
            auto topKSpecSize = sentencesCtx["topKSpec"].size();
            for (size_t i = 0; i < topKSpecSize; i++) {
                std::vector<uint32_t> v;
                for (size_t j = 0; j < sentencesCtx["topKSpec"][i].size(); j++) {
                    v.push_back(sentencesCtx["topKSpec"][i][j]);
                }
                tree.topKSpec.push_back(v);
            }
        }
        param.treeList.push_back(tree);
    } else {
        auto treeSize = sentencesCtx["specTree"].size();
        for (auto& specTree : sentencesCtx["specTree"]) {
            SpecTreeParam tree;
            tree.topHead.clear();
            tree.topKSpec.clear();
            tree.treeType = GetJsonValue(specTree, "treeType", HIAI_LLMENGINE_STATIC_SPEC_TREE);
            tree.gamma = GetJsonValue(specTree, "gamma", 3);
            tree.gammaThreshold = GetJsonValue(specTree, "gammaThreshold", 0.0);
            if (specTree.contains("topHead")) {
                auto topHeadSize = specTree["topHead"].size();
                for (size_t i = 0; i < topHeadSize; i++) {
                    tree.topHead.push_back(specTree["topHead"][i]);
                }
            }
            if (specTree.contains("topKSpec")) {
                auto topKSpecSize = specTree["topKSpec"].size();
                for (size_t i = 0; i < topKSpecSize; i++) {
                    std::vector<uint32_t> v;
                    for (size_t j = 0; j < specTree["topKSpec"][i].size(); j++) {
                        v.push_back(specTree["topKSpec"][i][j]);
                    }
                    tree.topKSpec.push_back(v);
                }
            }
            param.treeList.push_back(tree);
        }
    }
    param.gammaStrategy = GetJsonValue(sentencesCtx, "gammaStrategy", HIAI_LLMENGINE_GAMMA_DISABLE);
    param.emaWindow = GetJsonValue(sentencesCtx, "emaWindow", 1);
    param.typicalScaling = GetJsonValue(sentencesCtx, "typicalScaling", 1.0);
    param.sampleGreedy = GetJsonValue(sentencesCtx, "sampleGreedy", false);
    param.rejectScaling = GetJsonValue(sentencesCtx, "rejectScaling", false);
    param.draftThreshold = GetJsonValue(sentencesCtx, "draftThreshold", 0.05);
    param.specType = GetJsonValue(sentencesCtx, "specType", HIAI_LLMENGINE_TREE_SPECTYPE);
    param.draftText = GetJsonValue(sentencesCtx, "draftText", std::string(""));
    param.matchLen = GetJsonValue(sentencesCtx, "matchLen", 0);
    param.windowSize = GetJsonValue(sentencesCtx, "windowSize", 0);
}

int InitSentencesCtxParm(LLMEngineProcessParam& param, nlohmann::json& sentencesCtx)
{
    param.prompt = GetJsonValue(sentencesCtx, "prompt", std::string(""));
    auto tokenIdSize = sentencesCtx["tokenids"].size();
    for (int i = 0; i < tokenIdSize; i++) {
        param.tokenids.push_back(sentencesCtx["tokenids"][i]);
        LLM_ENGINE_BBIT_LOG_INFO("Generate tokenids: %d", param.tokenids[i]);
    }

    param.img = GetJsonValue(sentencesCtx, "img", std::vector<std::string>());
    param.imgEncoded = GetJsonValue(sentencesCtx, "imgEncoded", std::vector<std::string>());
    param.audioEncoded = GetJsonValue(sentencesCtx, "audioEncoded", std::vector<std::string>());
    param.audioEncodedSpec = GetJsonValue(sentencesCtx, "audioEncodedSpec", std::vector<std::string>());
    param.callbackFreq = GetJsonValue(sentencesCtx, "callbackFreq", 2);
    param.sampleFlag = GetJsonValue(sentencesCtx, "sampleFlag", false);
    param.seed = GetJsonValue(sentencesCtx, "seed", 99);
    param.topK = GetJsonValue(sentencesCtx, "topK", 1);
    param.topP = GetJsonValue(sentencesCtx, "topP", 1.0);
    param.temperature = GetJsonValue(sentencesCtx, "temperature", 1.0);
    param.maxGenTokens = GetJsonValue(sentencesCtx, "maxGenTokens", 99);
    param.repetitionPenalty = GetJsonValue(sentencesCtx, "repetitionPenalty", 1.0);
    param.initTokenLen = GetJsonValue(sentencesCtx, "initTokenLen", 0);
    param.isAsync = GetJsonValue(sentencesCtx, "isAsync", true);
    param.firstTokenAcclFlag = GetJsonValue(sentencesCtx, "firstTokenAcclFlag", false);
    param.setLatencyMs = GetJsonValue(sentencesCtx, "setLatencyMs", 10);
    param.firstTokenAcclMask = GetJsonValue(sentencesCtx, "firstTokenAcclMask", std::string(""));
    param.stopSeq.clear();
    if (sentencesCtx.contains("stopSeq")) {
        for (size_t i = 0; i < sentencesCtx["stopSeq"].size(); i++) {
            param.stopSeq.emplace_back(sentencesCtx["stopSeq"][i]);
        }
    }
    // prompt cache
    param.prefixPrompt = GetJsonValue(sentencesCtx, "prefixPrompt", std::string(""));
    param.pmtCacheOperation = GetJsonValue(sentencesCtx, "pmtCacheOperation", std::string(""));
    param.pmtCacheZeroCopy = GetJsonValue(sentencesCtx, "pmtCacheZeroCopy", false);
    param.kvcacheHandle = GetJsonValue(sentencesCtx, "kvcacheHandle", std::string(""));
    param.multiTurnFlag = GetJsonValue(sentencesCtx, "multiTurnFlag", false);
    if (param.multiTurnFlag) {
        param.prefixPrompt = g_historyPrompt + param.prefixPrompt;
    }
    // compress context
    param.compressRatio = GetJsonValue(sentencesCtx, "compressRatio", 0.4);
    param.addtionalContext = GetJsonValue(sentencesCtx, "addtionalContext", std::string(""));
    param.forceTokens.clear();
    if (sentencesCtx.contains("forceTokens")) {
        for (size_t i = 0; i < sentencesCtx["forceTokens"].size(); i++) {
            param.forceTokens.emplace_back(sentencesCtx["forceTokens"][i]);
        }
    }
    param.postPrompt = GetJsonValue(sentencesCtx, "postPrompt", std::string(""));
    param.prePosition = GetJsonValue(sentencesCtx, "prePosition", std::string(""));
    TreeSpecInferParam(param, sentencesCtx);
    param.inferPerfMode = GetJsonValue(sentencesCtx, "inferPerfMode",
        static_cast<int>(HIAI_LLMEngine_InferPerf_EXTREME_HIGH));
    param.updateLoraCfgPath = GetJsonValue(sentencesCtx, "updateLoraCfgPath", std::string(""));
    param.updateSpecLoraCfgPath = GetJsonValue(sentencesCtx, "updateSpecLoraCfgPath", std::string(""));
    return 0;
}

int InitLLMEngineProcessParam(LLMEngineProcessParam& param, nlohmann::json& testCaseCtx)
{
    param.inferType = GetJsonValue(testCaseCtx, "inferType", HIAI_LLMENGINE_UNDEFINED_INFERTYPE);
    param.tokenizerType = GetJsonValue(testCaseCtx, "tokenizerType", HIAI_LLMENGINE_UNDEFINED_TOKENIZER);
    param.tokenizerPath = GetJsonValue(testCaseCtx, "tokenizerPath", std::string("unknown path."));
    param.modelType = GetJsonValue(testCaseCtx, "modelType", HIAI_LLMENGINE_UNDEFINED_MODEL);
    param.modelPath = GetJsonValue(testCaseCtx, "modelPath", std::string("unknown path."));
    param.weightDir = GetJsonValue(testCaseCtx, "weightDir", std::string("unknown dir."));
    param.loraCfgPath = GetJsonValue(testCaseCtx, "loraCfgPath", std::string(""));
    param.baseLoraCfgPath = GetJsonValue(testCaseCtx, "baseLoraCfgPath", std::string(""));
    param.updateLoraFromBuffer = GetJsonValue(testCaseCtx, "updateLoraFromBuffer", false);
    if (param.inferType == HIAI_LLMENGINE_TREESPEC_INFERTYPE) {
        param.specModelType = GetJsonValue(testCaseCtx, "specModelType", HIAI_LLMENGINE_UNDEFINED_MODEL);
        param.specModelPath = GetJsonValue(testCaseCtx, "specModelPath", std::string("unknown path."));
        param.specWeightDir = GetJsonValue(testCaseCtx, "specWeightDir", std::string("unknown dir."));
        param.specLoraCfgPath = GetJsonValue(testCaseCtx, "specLoraCfgPath", std::string(""));
        param.baseSpecLoraCfgPath = GetJsonValue(testCaseCtx, "baseSpecLoraCfgPath", std::string(""));
        param.updateSpecLoraFromBuffer = GetJsonValue(testCaseCtx, "updateSpecLoraFromBuffer", false);
    }
    param.compressInferType = GetJsonValue(testCaseCtx, "compressInferType", HIAI_LLMENGINE_UNDEFINED_INFERTYPE);
    param.compressTokenizerType = GetJsonValue(testCaseCtx, "compressTokenizerType", HIAI_LLMENGINE_UNDEFINED_TOKENIZER);
    param.compressTokenizerPath = GetJsonValue(testCaseCtx, "compressTokenizerPath", std::string("unknown path."));
    param.compressModelType = GetJsonValue(testCaseCtx, "compressModelType", HIAI_LLMENGINE_UNDEFINED_MODEL);
    param.compressModelPath = GetJsonValue(testCaseCtx, "compressModelPath", std::string("unknown path."));
    param.compressWeightDir = GetJsonValue(testCaseCtx, "compressWeightDir", std::string("unknown dir."));
    param.preprocessorConfigPath = GetJsonValue(testCaseCtx, "preprocessorConfigPath", std::string(""));
    param.vitModelType = GetJsonValue(testCaseCtx, "vitModelType", HIAI_LLMENGINE_LLAMA2_MODEL);
    param.vitModelPath = GetJsonValue(testCaseCtx, "vitModelPath", std::string(""));
    param.vitWeightDir = GetJsonValue(testCaseCtx, "vitWeightDir", std::string(""));
    param.mlpModelPath = GetJsonValue(testCaseCtx, "mlpModelPath", std::string(""));
    param.mlpWeightDir = GetJsonValue(testCaseCtx, "mlpWeightDir", std::string(""));
    g_LLMEnginePrefillLen = GetJsonValue(testCaseCtx, "prefillLen", 32);
    param.isSwitchKVCache = GetJsonValue(testCaseCtx, "isSwitchKVCache", false);
    param.updateWeight = GetJsonValue(testCaseCtx, "updateWeight", true);
    return 0;
}

int RunImgEncode(LLMEngineProcessParam& param)
{
    if (param.vitModelPath.size() == 0) {
        return 0;
    }
    LLM_ENGINE_BBIT_LOG_INFO("Run Img Encode!!!");
    if (InitVisionProcess(param) != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("InitVisionProcess failed.");
        return -1;
    }
    LLM_ENGINE_BBIT_LOG_INFO("input add text.");

    for (size_t i = 0; i < g_imgPromptInfos.size(); i++) {
        if (!LoadImgs(g_imgPromptInfos[i].imgs, g_imgPrompt[i].imgBufs, g_imgPrompt[i].imgBufsSizes)) {
            LLM_ENGINE_BBIT_LOG_ERROR("LoadImgs failed.");
            return false;
        }
        if (!g_imgPrompt[i].imgBufs.empty()) {
            LLM_ENGINE_BBIT_LOG_INFO("input add img.");
            if (HIAI_LLMEngine_Prompt_SetImageBufs(g_mllmPrompts[i], g_imgPrompt[i].imgBufs.data(),
                g_imgPrompt[i].imgBufsSizes.data(), g_imgPrompt[i].imgBufs.size()) != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetImageBufs failed.");
                return false;
            }
        }
        if (!LoadImgs(g_imgPromptInfos[i].imgEncodeds, g_imgPrompt[i].imgEncodedBufs, g_imgPrompt[i].imgEncodedBufsSizes)) {
            LLM_ENGINE_BBIT_LOG_ERROR("LoadImgs Encoded failed.");
            return false;
        }
        if (!g_imgPrompt[i].imgEncodedBufs.empty()) {
            LLM_ENGINE_BBIT_LOG_INFO("input encoded add img.");
            if (HIAI_LLMEngine_Prompt_SetEncodedImageBufs(g_mllmPrompts[i], g_imgPrompt[i].imgEncodedBufs.data(),
                g_imgPrompt[i].imgEncodedBufsSizes.data(), g_imgPrompt[i].imgEncodedBufs.size()) != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetEncodedImageBufs failed.");
                return false;
            }
        }

        if (!g_imgPrompt[i].imgInfos.empty()) {
            LLM_ENGINE_BBIT_LOG_INFO("input imgInfos.");
            if (HIAI_LLMEngine_Prompt_SetImageInfosV2(g_mllmPrompts[i], g_imgPrompt[i].imgInfos.data(), g_imgPrompt[i].imgInfos.size()) != hiai::SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetImageInfosV2 failed.");
                return false;
            }
        }
    }

    auto ret = HIAI_LLMEngine_Executor_EncodePrompts(g_LLMEngineExecutor, g_mllmPrompts.data(), g_mllmPrompts.size());
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_EncodePrompts failed.");
        return -1;
    }
    LLMEngineVisionDeInit();
    for (size_t i = 0; i < g_imgPromptInfos.size(); i++) {
        for (auto& imgBuf: g_imgPrompt[i].imgBufs) {
            delete[] reinterpret_cast<uint8_t*>(imgBuf);
        }
        for (auto& imgEncodedBuf: g_imgPrompt[i].imgEncodedBufs) {
            delete[] reinterpret_cast<uint8_t*>(imgEncodedBuf);
        }
    }
    return 0;
}

int RunPromptCompress(LLMEngineProcessParam& param, nlohmann::json& sentencesCtx)
{
    // run prompt compress
    if (param.compressInferType == HIAI_LLMENGINE_LLMLINGUA_INFERTYPE) {
        if (LLMEngineInitCompress(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineInitCompress failed.");
            return -1;
        }
        if (LLMEngineCompress(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineRun failed.");
            return -1;
        }
        if (LLMEngineDeInitCompress(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineDeInitCompress failed.");
            return -1;
        }
        sentencesCtx["compressed_prompt"] = param.prompt;
    }
    return 0;
}

int ProcPromptKVCache(LLMEngineProcessParam& param, size_t& cacheSize, std::string& saveFilePrefix,
    HIAI_LLMEngine_PromptKVCache* promptCacheSave, HIAI_LLMEngine_PromptKVCache* promptCacheLoad)
{
    if (param.isSwitchKVCache) {
        return 0;
    }
    if (param.pmtCacheOperation == "load") {
        LLM_ENGINE_BBIT_LOG_INFO("#### Create PromptKVCache from Buffer.");
        if (cacheSize == 0) return -1;
        uint8_t* cacheAddrLoad = new uint8_t[cacheSize];
        hiai::ScopeGuard promptCacheLoadGuard([&] () {
            if (promptCacheLoad != nullptr) {
                HIAI_LLMEngine_PromptKVCache_Destory(&promptCacheLoad);
                LLM_ENGINE_BBIT_LOG_INFO("promptCacheLoad has been released");
            }
            
            delete[] cacheAddrLoad;
            cacheAddrLoad = nullptr;
            LLM_ENGINE_BBIT_LOG_INFO("cacheAddrLoad has been released");
        });
        HIAI_EXPECT_TRUE(LoadBinaryFile(saveFilePrefix, reinterpret_cast<void*>(cacheAddrLoad), cacheSize));
        promptCacheLoad = HIAI_LLMEngine_PromptKVCache_CreateFromBuffer(reinterpret_cast<void*>(cacheAddrLoad), cacheSize);
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Executor_LoadPromptKVCache(g_LLMEngineExecutor, g_LLMEngineContext, promptCacheLoad)
            == hiai::SUCCESS, -1);
        if (!param.updateLoraCfgPath.empty()) {
            LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora config file path: %s", param.updateLoraCfgPath.c_str());
            auto ret = HIAI_LLMEngine_Executor_UpdateLora(g_LLMEngineExecutor, param.updateLoraCfgPath.c_str());
            if (ret != HIAI_LLMEngine_SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateLora failed.");
                return -1;
            }
            LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora Done.");
        }
        if (!param.updateSpecLoraCfgPath.empty()) {
            LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora config file path: %s", param.updateSpecLoraCfgPath.c_str());
            auto ret = HIAI_LLMEngine_Executor_UpdateSpecLora(g_LLMEngineExecutor, param.updateSpecLoraCfgPath.c_str());
            if (ret != HIAI_LLMEngine_SUCCESS) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateSpecLora failed.");
                return -1;
            }
            LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora Done.");
        }
    } else if (param.pmtCacheOperation == "save") {
        LLM_ENGINE_BBIT_LOG_INFO("#### Compute PromptKVCache and Copy.");
        promptCacheSave = HIAI_LLMEngine_PromptKVCache_Create();
        hiai::ScopeGuard promptCacheSaveGuard([&] () {
            HIAI_LLMEngine_PromptKVCache_Destory(&promptCacheSave);
            LLM_ENGINE_BBIT_LOG_INFO("promptCacheSave has been released");
        });
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Executor_SavePromptKVCache(g_LLMEngineExecutor, g_LLMEngineContext, promptCacheSave)
            == hiai::SUCCESS, -1);
        void* cacheAddrSave = nullptr;
        HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_PromptKVCache_GetData(promptCacheSave, &cacheAddrSave, &cacheSize)
            == hiai::SUCCESS, -1);
        LLM_ENGINE_BBIT_LOG_INFO("#### write cacheSize is  %lu.", cacheSize);
        LLM_ENGINE_BBIT_LOG_INFO("#### write cacheAddr is  %p.", cacheAddrSave);
        saveFilePrefix = saveFilePrefix + std::to_string(cacheSize) + ".bin";
        DumpBinaryToFile(saveFilePrefix, cacheAddrSave, cacheSize);
    }
    return 0;
}

int UpdateLora(LLMEngineProcessParam &param)
{
    if (!param.updateLoraCfgPath.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora config file path: %s", param.updateLoraCfgPath.c_str());
        auto ret = HIAI_LLMEngine_Executor_UpdateLora(g_LLMEngineExecutor, param.updateLoraCfgPath.c_str());
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateLora failed.");
            return -1;
        }
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update lora Done.");
    }
    if (!param.updateSpecLoraCfgPath.empty()) {
        LLM_ENGINE_BBIT_LOG_INFO(
            "LLM Engine, Update spec lora config file path: %s", param.updateSpecLoraCfgPath.c_str());
        auto ret = HIAI_LLMEngine_Executor_UpdateSpecLora(g_LLMEngineExecutor, param.updateSpecLoraCfgPath.c_str());
        if (ret != HIAI_LLMEngine_SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_UpdateSpecLora failed.");
            return -1;
        }
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Update spec lora Done.");
    }
    return 0;
}

int SaveKVCache(LLMEngineProcessParam& param, HIAI_LLMEngine_KVCache* curKVCache, size_t sentencesCtx_num, std::string& saveFilePrefix)
{
    int64_t maxGenTokens = param.maxGenTokens;
    param.maxGenTokens = 0;
    if (LLMEngineRun(param, sentencesCtx_num) != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineRun failed.");
        return -1;
    }
    param.maxGenTokens = maxGenTokens;

    if (HIAI_LLMEngine_KVCache_GetRealSize(curKVCache, &g_size) != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_KVCache_GetRealSize failed.");
        return -1;
    }
    if (g_size == 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("g_size is 0.");
        return -1;
    }
    g_data = reinterpret_cast<void*>(new (std::nothrow) uint8_t[g_size]);
    if (HIAI_LLMEngine_KVCache_SaveToBuffer(curKVCache, g_data, g_size) != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_KVCache_SaveToBuffer failed.");
        return -1;
    }
    if (param.pmtCacheZeroCopy) {
        LLM_ENGINE_BBIT_LOG_INFO("write cacheSize is %lu.", g_size);
        saveFilePrefix = saveFilePrefix + std::to_string(g_size) + ".bin";
        DumpBinaryToFile(saveFilePrefix, g_data, g_size);
        delete[] reinterpret_cast<uint8_t*>(g_data);
        g_data = nullptr;
    }
    LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, save to buffer Done.");
    return 0;
}

int LoadKVCache(LLMEngineProcessParam& param, HIAI_LLMEngine_KVCache* curKVCache, std::string& saveFilePrefix)
{
    if (param.pmtCacheZeroCopy) {
        if (HIAI_LLMEngine_KVCache_LoadFromFile(curKVCache, saveFilePrefix.c_str()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_KVCache_LoadFromBuffer failed.");
            return -1;
        }
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Load from file Done.");
    } else {
        if (HIAI_LLMEngine_KVCache_LoadFromBuffer(curKVCache, g_data, g_size) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_KVCache_LoadFromBuffer failed.");
            return -1;
        }
        LLM_ENGINE_BBIT_LOG_INFO("LLM Engine, Load from buffer Done.");
        uint8_t* cacheAddr = reinterpret_cast<uint8_t*>(g_data);
        delete[] cacheAddr;
    }
    if (UpdateLora(param) != 0) {
        return -1;
    }
    return 0;
}

int SetKVCache(LLMEngineProcessParam& param, size_t sentencesCtx_num, std::string& saveFilePrefix)
{
    if (!param.isSwitchKVCache) {
        return 0;
    }

    HIAI_LLMEngine_KVCache* curKVCache = nullptr;
    if (param.kvcacheHandle == "session1") {
        if (g_KVCache1 == nullptr) {
            g_KVCache1 = HIAI_LLMEngine_KVCache_Create(g_LLMEngineExecutor);
            if (g_KVCache1 == nullptr) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_KVCache_Create failed.");
                return -1;
            }
        }
        if (HIAI_LLMEngine_Context_SetKVCache(g_LLMEngineContext, g_KVCache1) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetKVCache failed.");
            return -1;
        }
        curKVCache = g_KVCache1;
    } else if (param.kvcacheHandle == "session2") {
        if (g_KVCache2 == nullptr) {
            g_KVCache2 = HIAI_LLMEngine_KVCache_Create(g_LLMEngineExecutor);
            if (g_KVCache2 == nullptr) {
                LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_KVCache_Create failed.");
                return -1;
            }
        }

        if (HIAI_LLMEngine_Context_SetKVCache(g_LLMEngineContext, g_KVCache2) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_SetKVCache failed.");
            return -1;
        }
        curKVCache = g_KVCache2;
    }

    if (param.pmtCacheOperation == "save") {
        if (SaveKVCache(param, curKVCache, sentencesCtx_num, saveFilePrefix) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_SaveKVCache failed.");
            return -1;
        }
    } else if (param.pmtCacheOperation == "load") {
        if (LoadKVCache(param, curKVCache, saveFilePrefix) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_LoadKVCache failed.");
            return -1;
        }
    }
    return 0;
}

int LLMEngineOutputGenerate()
{
    int ret = -1;
    if (g_textPrompt.size() > 0) {
        uint32_t len = 0;
        ret = HIAI_LLMEngine_Context_GetAllGenerationLen(g_LLMEngineContext, &len);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGenerationLen failed.");
            return false;
        }

        char* generation = new char[len + 1];
        ret = HIAI_LLMEngine_Context_GetAllGeneration(g_LLMEngineContext, generation, len);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGeneration failed.");
            return false;
        }
        generation[len] = '\0';
        LLM_ENGINE_BBIT_LOG_INFO("all generation len: %d.", len);
        LLM_ENGINE_BBIT_LOG_INFO("all generation: %s.", generation);
        delete[] generation;
    } else if (g_tokenidPrompt.size() > 0) {
        uint32_t genTokenlen = 0;
        ret = HIAI_LLMEngine_Context_GetAllTokenGenerationLen(g_LLMEngineContext, &genTokenlen);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetAllGenerationLen failed.");
            return false;
        }

        int32_t* genToken = new int32_t[genTokenlen + 1];
        auto ret = HIAI_LLMEngine_Context_GetAllTokenGeneration(g_LLMEngineContext, genToken, genTokenlen);
        if (ret != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOneTokenGeneration failed.");
            return false;
        }

        genToken[genTokenlen] = '\0';
        LLM_ENGINE_BBIT_LOG_INFO("all tokenid len: %d.", genTokenlen);
        for (int i = 0; i < genTokenlen; i++) {
            LLM_ENGINE_BBIT_LOG_INFO("tokenid %d: %d.", i, genToken[i]);
        }
        delete[] genToken;
    }

    double totalTimeMs = 0;
    ret = HIAI_LLMEngine_Context_GetTotalTimeMs(g_LLMEngineContext, &totalTimeMs);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetTotalTimeMs failed.");
        return true;
    }
    g_LLMEngineTotalTimeMs = totalTimeMs;

    double prefillTimeMs = 0;
    ret = HIAI_LLMEngine_Context_GetPrefillTimeMs(g_LLMEngineContext, &prefillTimeMs);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetPrefillTimeMs failed.");
        return true;
    }
    g_LLMEnginePrefillTimeMs = prefillTimeMs;

    double decodeTimeMs = 0;
    ret = HIAI_LLMEngine_Context_GetDecodeTimeMs(g_LLMEngineContext, &decodeTimeMs);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetDecodeTimeMs failed.");
        return true;
    }
    g_LLMEngineDecodeTimeMs = decodeTimeMs;

    uint64_t inputTokenCount = 0;
    ret = HIAI_LLMEngine_Context_GetInputTokenCount(g_LLMEngineContext, &inputTokenCount);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetInputTokenCount failed.");
        return true;
    }
    g_LLMEngineInputTokenCount = inputTokenCount;

    uint64_t outputTokenCount = 0;
    ret = HIAI_LLMEngine_Context_GetOutputTokenCount(g_LLMEngineContext, &outputTokenCount);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetOutputTokenCount failed.");
        return true;
    }
    g_LLMEngineOutputTokenCount = outputTokenCount;

    uint64_t cachedTokenCount = 0;
    ret = HIAI_LLMEngine_Context_GetCachedTokenCount(g_LLMEngineContext, &cachedTokenCount);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetCachedTokenCount failed.");
        return true;
    }
    g_LLMEngineCachedTokenCount = cachedTokenCount;

    uint64_t decodeNum = 0;
    ret = HIAI_LLMEngine_Context_GetDecodeNum(g_LLMEngineContext, &decodeNum);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Context_GetDecodeNum failed.");
        return true;
    }
    g_LLMEngineDecodeNum = decodeNum;

    g_LLMEngineDecodePerTokenTimeMs = decodeTimeMs / outputTokenCount;
    g_LLMEngineDecodePerBlockTimeMs = decodeTimeMs / g_LLMEngineDecodeNum;

    LLM_ENGINE_BBIT_LOG_INFO("inputTokenCount: %lu.", inputTokenCount);
    LLM_ENGINE_BBIT_LOG_INFO("outputTokenCount: %lu.", outputTokenCount);
    LLM_ENGINE_BBIT_LOG_INFO("cachedTokenCount: %lu.", cachedTokenCount);
    LLM_ENGINE_BBIT_LOG_INFO("decodeNum: %lu.", decodeNum);
    LLM_ENGINE_BBIT_LOG_INFO("totalTimeMs: %.5f ms.", totalTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("prefillTimeMs: %.5f ms.", prefillTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("decodeTimeMs: %.5f ms.", decodeTimeMs);
    LLM_ENGINE_BBIT_LOG_INFO("decodeTimeMs per token: %.5f ms.", decodeTimeMs / outputTokenCount);
    LLM_ENGINE_BBIT_LOG_INFO("prefillLen: %lu.", g_LLMEnginePrefillLen);
    g_LLMEngineWaitFlag = false;
    g_LLMEnginePrefillPerInferTimeMs = prefillTimeMs / ceil(inputTokenCount / g_LLMEnginePrefillLen);
    return true;
}

int LLMEnginePromptsGenerate(const char* json_path)
{
    nlohmann::json jsonPrompt;
    try {
        std::ifstream(json_path) >> jsonPrompt;
    } catch (const std::exception e) {
        LLM_ENGINE_BBIT_LOG_ERROR("parse json failed, please check json file: %s.", json_path);
        return -1;
    }

    HIAI_LLMEngine_Prompt* mllmPrompt = HIAI_LLMEngine_Prompt_Create();
    if (mllmPrompt == nullptr) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_Create fail.");
        return -1;
    }

    if (jsonPrompt.contains("text")) {
        g_textPrompt.clear();
        g_textPrompt = GetJsonValue(jsonPrompt, "text", std::string(""));
        LLM_ENGINE_BBIT_LOG_INFO("input add text:%s", g_textPrompt.c_str());
    }

    if (jsonPrompt.contains("tokenids")) {
        LLM_ENGINE_BBIT_LOG_INFO("input add tokenids.");
        g_tokenidPrompt.clear();
        auto tokenIdSize = jsonPrompt["tokenids"].size();
        for (int i = 0; i < tokenIdSize; i++) {
            g_tokenidPrompt.push_back(jsonPrompt["tokenids"][i]);
            LLM_ENGINE_BBIT_LOG_INFO("Generate tokenids: %d", g_tokenidPrompt[i]);
        }
    }

    // load text
    if (g_textPrompt.size() > 0) {
        LLM_ENGINE_BBIT_LOG_INFO("input add text.");
        if (HIAI_LLMEngine_Prompt_SetText(mllmPrompt, g_textPrompt.c_str()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetText failed.");
            return false;
        }
    }

    // load tokenid
    if (g_tokenidPrompt.size() > 0) {
        LLM_ENGINE_BBIT_LOG_INFO("input add tokenid.");
        if (HIAI_LLMEngine_Prompt_SetTokenIds(mllmPrompt, g_tokenidPrompt.data(),
            g_tokenidPrompt.size()) != hiai::SUCCESS) {
            LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Prompt_SetText failed.");
            return false;
        }
    }

    // run
    LLM_ENGINE_BBIT_LOG_INFO("HIAI_LLMEngine_Executor_MLLMGenerate");
    int ret = HIAI_LLMEngine_Executor_MLLMGenerate(g_LLMEngineExecutor, g_LLMEngineContext, mllmPrompt);
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_Generate failed.");
        return false;
    }

    ret = LLMEngineOutputGenerate();
    if (ret != hiai::SUCCESS) {
        LLM_ENGINE_BBIT_LOG_ERROR("HIAI_LLMEngine_Executor_Generate failed.");
        return false;
    }
    return 0;
}

int LLMEngineContextGenerate(const char* json_path)
{
    FILE* file = fopen(json_path, "r");
    if (file == nullptr) {
        printf("Failed to open JSON file!\n");
        return -1;
    }

    // 读取文件内容到内存
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* jsonStr = (reinterpret_cast <char*>(malloc(file_size + 1)));
    if (jsonStr == nullptr) {
        printf("Failed to allocate memory for JSON string!\n");
        fclose(file);
        return -1;
    }
    fread(jsonStr, 1, file_size, file);
    fclose(file);
    jsonStr[file_size] = '\0';  // 确保字符串以空字符结尾

    g_LLMEngineContext = HIAI_LLMEngine_Context_CreateFromContextJson(jsonStr);
    if (g_LLMEngineContext == nullptr) {
        printf("Failed to create context from JSON!\n");
        free(jsonStr);
        return -1;
    }
    free(jsonStr);
    return 0;
}

int LLMEngineExecutorGenerate(const char* json_path)
{
    FILE* file = fopen(json_path, "r");
    if (file == nullptr) {
        printf("Failed to open JSON file!\n");
        return -1;
    }

    // 读取文件内容到内存
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char* jsonStr = (reinterpret_cast <char*>(malloc(file_size + 1)));
    if (jsonStr == nullptr) {
        printf("Failed to allocate memory for JSON string!\n");
        fclose(file);
        return -1;
    }
    fread(jsonStr, 1, file_size, file);
    fclose(file);
    jsonStr[file_size] = '\0';  // 确保字符串以空字符结尾

    g_LLMEngineExecutor = HIAI_LLMEngine_Executor_CreateFromJson(jsonStr);
    if (g_LLMEngineExecutor == nullptr) {
        printf("Failed to create exexutor from JSON!\n");
        free(jsonStr);
        return -1;
    }
    free(jsonStr);
    return 0;
}

int LLMEngineGenerateFromJson(char* argv[])
{
    LLM_ENGINE_BBIT_LOG_INFO("## HIAI_LLMEngine_Context_CreateFromContextJson, start...");

    const char* contextjson_path = argv[1];
    if (LLMEngineContextGenerate(contextjson_path) != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineContextGenerate failed.");
        return -1;
    }

    LLM_ENGINE_BBIT_LOG_INFO("## HIAI_LLMEngine_Executor_CreateFromJson, start...");
    const char* exectorjson_path = argv[2];
    if (LLMEngineExecutorGenerate(exectorjson_path) != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineExecutorGenerate failed.");
        return -1;
    }

    const char* promptjson_path = argv[3];
    if (LLMEnginePromptsGenerate(promptjson_path) != 0) {
        LLM_ENGINE_BBIT_LOG_ERROR("LLMEnginePromptsGenerate failed.");
        return -1;
    }

    LLMEngineDeInit();
    return 0;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    if (argc < 2) {
        LLM_ENGINE_BBIT_LOG_ERROR("you should input json file path.");
        return -1;
    }

    if (argc == 4 && std::string(argv[1]) == "context.json" && std::string(argv[2]) == "executor.json" &&
        std::string(argv[3]) == "prompt.json") {
        LLM_ENGINE_BBIT_LOG_INFO("##argc = %d, argv[0] = %s, argv[1] = %s, argv[2] = %s, argv[3] = %s",
            argc, argv[0], argv[1], argv[2], argv[3]);
        LLM_ENGINE_BBIT_LOG_INFO("## LLMEngineGenerateFromJson, start...");
        int ret = LLMEngineGenerateFromJson(argv);
        LLM_ENGINE_BBIT_LOG_INFO("LLMEngineGenerateFromJson return %d", ret);
        return ret;
    }

    nlohmann::json jsonContext;
    try {
        std::ifstream(argv[1]) >> jsonContext;
    } catch (const std::exception e) {
        LLM_ENGINE_BBIT_LOG_ERROR("parse json failed, please check json file: %s.", argv[1]);
        return -1;
    }
    HIAI_LLMEngine_PromptKVCache* promptCacheSave = nullptr;
    HIAI_LLMEngine_PromptKVCache* promptCacheLoad = nullptr;
    size_t cacheSize = 0;
    std::string saveFilePrefix = "PromptKVCache";
    for (auto& testCaseCtx: jsonContext) {
        LLMEngineProcessParam param;
        const std::string testCaseName = GetJsonValue(testCaseCtx, "testCaseName", std::string("unknown name."));
        LLM_ENGINE_BBIT_LOG_INFO("## testCaseName: %s, start...", testCaseName.c_str());
        if (InitLLMEngineProcessParam(param, testCaseCtx) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineExecutorInit failed.");
            return -1;
        }
        nlohmann::json maintainCtx = GetJsonValue(testCaseCtx, "maintain", nlohmann::json({}));
        std::string FLAGS_enable_item = GetJsonValue(maintainCtx, "FLAGS_enable_item", std::string("0"));
        std::string configStr = GetJsonValue(maintainCtx, "configStr", std::string(""));
        std::string romVersion = GetJsonValue(maintainCtx, "romVersion", std::string(""));
        std::shared_ptr<hiai::HiAiMaintWrapper> maintClient = std::make_shared<hiai::HiAiMaintWrapper>();
        if (FLAGS_enable_item != "0") {
            if (maintClient->EnableProcessing(FLAGS_enable_item, romVersion, configStr) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("[ModelManagerV2]Enable dump fail.\n");
                return -1;
            }
        }
        ClearCache();
        if (!testCaseCtx.contains("sentences")) {
            LLM_ENGINE_BBIT_LOG_ERROR("not contains sentences.");
            return -1;
        }
        if (LLMEngineExecutorInit(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineExecutorInit failed.");
            return -1;
        }
        if (LLMEnginePromptsInit(testCaseCtx["sentences"]) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEnginePromptsInit failed.");
            return -1;
        }

        LLM_ENGINE_BBIT_LOG_INFO("RunVit");
        if (RunImgEncode(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("SetPrompts failed.");
            return -1;
        }
        if (LLMEngineInit(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineInit failed.");
            return -1;
        }
        if (LLMEngineUpdateLora(param) != 0) {
            LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineUpdateLora failed.");
            return -1;
        }
        size_t sentencesCtx_num = 0;
        for (auto& sentencesCtx: testCaseCtx["sentences"]) {
            if (InitSentencesCtxParm(param, sentencesCtx) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("LLMEnginePromptsInit failed.");
                return -1;
            }
            InitTestInfo();

            HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetPrefixPrompt(g_LLMEngineContext, param.prefixPrompt.c_str()) ==
                hiai::SUCCESS, -1);
            HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_SetInitTokenLen(g_LLMEngineContext, param.initTokenLen) ==
                hiai::SUCCESS, -1);
            if (ProcPromptKVCache(param, cacheSize, saveFilePrefix, promptCacheSave, promptCacheLoad) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("ProcPromptKVCache failed.");
                return -1;
            }
            if (SetKVCache(param, sentencesCtx_num, saveFilePrefix) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("SetKVCache failed.");
                return -1;
            }
            if (RunPromptCompress(param, sentencesCtx) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineRun failed.");
                return -1;
            }

            if (LLMEngineRun(param, sentencesCtx_num) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("LLMEngineRun failed.");
                return -1;
            }
            // multiturn
            if (param.multiTurnFlag) {
                g_historyPrompt = param.prefixPrompt + param.prompt + g_LLMEngineGeneration + param.stopSeq[0];
            } else {
                g_historyPrompt = "";
            }
            LLM_ENGINE_BBIT_LOG_INFO("expect: %s", std::string(sentencesCtx["expect"]).c_str());
            // write result
            sentencesCtx["generation"] = g_LLMEngineGeneration;
            sentencesCtx["totalTimeMs"] = g_LLMEngineTotalTimeMs;
            sentencesCtx["initTimeMs"] = g_LLMEngineInitTimeMs;
            sentencesCtx["prefillTimeMs"] = g_LLMEnginePrefillTimeMs;
            sentencesCtx["decodeTimeMs"] = g_LLMEngineDecodeTimeMs;
            sentencesCtx["prefillPerTokenTimeMs"] = g_LLMEnginePrefillPerInferTimeMs;
            sentencesCtx["decodePerTokenTimeMs"] = g_LLMEngineDecodePerTokenTimeMs;
            sentencesCtx["decodePerBlockTimeMs"] = g_LLMEngineDecodePerBlockTimeMs;
            sentencesCtx["InputTokenCount"] = g_LLMEngineInputTokenCount;
            sentencesCtx["OutputTokenCount"] = g_LLMEngineOutputTokenCount;
            sentencesCtx["CachedTokenCount"] = g_LLMEngineCachedTokenCount;
            sentencesCtx["DecodeNum"] = g_LLMEngineDecodeNum;
            sentencesCtx["block_efficiency"] = static_cast<double>(g_LLMEngineOutputTokenCount) /
                static_cast<double>(g_LLMEngineDecodeNum);
            UpdateTestInfo();
            HIAI_EXPECT_TRUE_R(HIAI_LLMEngine_Context_UnSetPrefixPrompt(g_LLMEngineContext) == hiai::SUCCESS, -1);
            sentencesCtx_num++;
        }
        LLMEngineDeInit();
        LLM_ENGINE_BBIT_LOG_INFO("## testCaseName: %s, Done", testCaseName.c_str());
        PrintTimeInfo();

        if (FLAGS_enable_item != "0") {
            usleep(25 * 2 * 1000);
            if (maintClient->DisableProcessing(FLAGS_enable_item, configStr) != 0) {
                LLM_ENGINE_BBIT_LOG_ERROR("[ModelManagerV2]DisableProcessing failed.\n");
                return -1;
            }
        }
    }
    if (argc >= 3) {
        LLM_ENGINE_BBIT_LOG_INFO("#### write all result in %s.", argv[2]);
        std::ofstream(argv[2]) << jsonContext.dump(4);
    }
    return 0;
}
