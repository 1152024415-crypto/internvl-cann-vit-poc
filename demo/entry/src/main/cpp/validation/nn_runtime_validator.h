#pragma once

#include "napi/native_api.h"
#include "rawfile/raw_file_manager.h"

#include <cstddef>
#include <string>
#include <vector>

namespace internvl {

struct RunResult {
    bool ok = false;
    std::string caseName;
    std::string deviceType;
    std::string errorStage;
    std::string errorMessage;
    double latencyMs = 0.0;
    std::size_t outputElementCount = 0;
    double maxAbsDiff = 0.0;
    double meanAbsDiff = 0.0;
    double cosine = 0.0;
    bool finite = false;
    bool outputShapeOk = false;
    std::string outputShape;
};

struct StabilityResult {
    bool ok = false;
    std::string caseName;
    int repeatCount = 0;
    int successCount = 0;
    double minLatencyMs = 0.0;
    double maxLatencyMs = 0.0;
    double avgLatencyMs = 0.0;
    std::string errorStage;
    std::string errorMessage;
    RunResult lastRun;
};

struct ModelStatusResult {
    bool ok = false;
    bool loaded = false;
    std::string deviceType;
    std::string errorStage;
    std::string errorMessage;
    double latencyMs = 0.0;
};

struct OfficialSmokeResult {
    bool ok = false;
    std::string deviceType;
    std::string errorStage;
    std::string errorMessage;
    double latencyMs = 0.0;
    std::size_t modelByteCount = 0;
    std::size_t inputCount = 0;
    std::size_t outputCount = 0;
    std::size_t inputByteCount = 0;
    std::size_t outputByteCount = 0;
};

std::vector<std::string> ListTestCaseNames();
ModelStatusResult LoadModel(NativeResourceManager* resourceManager);
ModelStatusResult LoadModel(napi_env env, napi_value resourceManager);
ModelStatusResult UnloadModel();
bool IsModelLoaded();
OfficialSmokeResult RunOfficialSmoke(NativeResourceManager* resourceManager);
OfficialSmokeResult RunOfficialSmoke(napi_env env, napi_value resourceManager);
RunResult RunOnce(NativeResourceManager* resourceManager, const std::string& caseName);
RunResult RunOnce(napi_env env, napi_value resourceManager, const std::string& caseName);
StabilityResult RunStability(NativeResourceManager* resourceManager, const std::string& caseName, int repeatCount);
StabilityResult RunStability(napi_env env, napi_value resourceManager, const std::string& caseName, int repeatCount);

} // namespace internvl
