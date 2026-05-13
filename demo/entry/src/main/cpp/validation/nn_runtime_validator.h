#pragma once

#include "napi/native_api.h"

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

std::vector<std::string> ListTestCaseNames();
RunResult RunOnce(napi_env env, napi_value resourceManager, const std::string& caseName);
StabilityResult RunStability(napi_env env, napi_value resourceManager, const std::string& caseName, int repeatCount);

} // namespace internvl
