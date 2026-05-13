#include "nn_runtime_validator.h"

#include "rawfile_loader.h"
#include "tensor_metrics.h"
#include "validation_constants.h"

#include "neural_network_runtime/neural_network_core.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>

namespace internvl {

namespace {

const char* kDeviceType = "HIAI_F";

const TestCase* FindCase(const std::string& caseName)
{
    for (const auto& testCase : kTestCases) {
        if (caseName == testCase.name) {
            return &testCase;
        }
    }
    return nullptr;
}

RunResult Fail(const std::string& caseName, const std::string& stage, const std::string& message)
{
    RunResult result;
    result.caseName = caseName;
    result.errorStage = stage;
    result.errorMessage = message;
    return result;
}

void DestroyExecutor(OH_NNExecutor** executor)
{
    if (executor != nullptr && *executor != nullptr) {
        OH_NNExecutor_Destroy(executor);
    }
}

void DestroyCompilation(OH_NNCompilation** compilation)
{
    if (compilation != nullptr && *compilation != nullptr) {
        OH_NNCompilation_Destroy(compilation);
    }
}

RunResult RunOfflineModel(const std::string& caseName,
                          const std::vector<std::uint8_t>& modelBytes,
                          const std::vector<std::uint8_t>& inputBytes,
                          std::vector<float>& output,
                          double& latencyMs)
{
    OH_NNCompilation* compilation = nullptr;
    OH_NNExecutor* executor = nullptr;

    // Yellow-zone note: keep all HarmonyOS NN calls in this file. If local SDK
    // headers expose slightly different signatures, patch this function only.
    const auto start = std::chrono::steady_clock::now();
    OH_NN_ReturnCode code = OH_NNCompilation_ConstructWithOfflineModelBuffer(
        modelBytes.data(),
        modelBytes.size(),
        &compilation);
    if (code != OH_NN_SUCCESS || compilation == nullptr) {
        return Fail(caseName, "om_compilation_failed", "OH_NNCompilation_ConstructWithOfflineModelBuffer failed");
    }

    code = OH_NNCompilation_SetDevice(compilation, kDeviceType);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return Fail(caseName, "device_selection_failed", "OH_NNCompilation_SetDevice(HIAI_F) failed");
    }

    code = OH_NNCompilation_Build(compilation);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return Fail(caseName, "om_compilation_failed", "OH_NNCompilation_Build failed");
    }

    code = OH_NNExecutor_Construct(compilation, &executor);
    if (code != OH_NN_SUCCESS || executor == nullptr) {
        DestroyCompilation(&compilation);
        return Fail(caseName, "executor_create_failed", "OH_NNExecutor_Construct failed");
    }

    code = OH_NNExecutor_SetInput(executor, 0, inputBytes.data(), inputBytes.size());
    if (code != OH_NN_SUCCESS) {
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "input_size_mismatch", "OH_NNExecutor_SetInput failed");
    }

    code = OH_NNExecutor_SetOutput(executor, 0, output.data(), kOutputByteCount);
    if (code != OH_NN_SUCCESS) {
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "output_size_mismatch", "OH_NNExecutor_SetOutput failed");
    }

    code = OH_NNExecutor_RunSync(executor);

    DestroyExecutor(&executor);
    DestroyCompilation(&compilation);

    const auto end = std::chrono::steady_clock::now();
    latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    if (code != OH_NN_SUCCESS) {
        return Fail(caseName, "run_sync_failed", "OH_NNExecutor_RunSync failed");
    }

    RunResult result;
    result.ok = true;
    result.caseName = caseName;
    result.deviceType = kDeviceType;
    result.latencyMs = latencyMs;
    result.outputElementCount = output.size();
    return result;
}

} // namespace

std::vector<std::string> ListTestCaseNames()
{
    std::vector<std::string> names;
    for (const auto& testCase : kTestCases) {
        names.emplace_back(testCase.name);
    }
    return names;
}

RunResult RunOnce(napi_env env, napi_value resourceManager, const std::string& caseName)
{
    const TestCase* testCase = FindCase(caseName);
    if (testCase == nullptr) {
        return Fail(caseName, "missing_artifact", "Unknown validation test case");
    }

    auto modelBytes = ReadRawfile(env, resourceManager, kModelFile);
    if (!modelBytes.ok) {
        return Fail(caseName, modelBytes.errorStage, modelBytes.errorMessage);
    }

    auto inputBytes = ReadRawfile(env, resourceManager, testCase->inputFile);
    if (!inputBytes.ok) {
        return Fail(caseName, inputBytes.errorStage, inputBytes.errorMessage);
    }
    if (inputBytes.bytes.size() != kInputByteCount) {
        return Fail(caseName, "input_size_mismatch", "Input rawfile byte count does not match [1,3,448,448] fp32");
    }

    auto expectedBytes = ReadRawfile(env, resourceManager, testCase->expectedOutputFile);
    if (!expectedBytes.ok) {
        return Fail(caseName, expectedBytes.errorStage, expectedBytes.errorMessage);
    }
    if (expectedBytes.bytes.size() != kOutputByteCount) {
        return Fail(caseName, "output_size_mismatch",
                    "Expected output rawfile byte count does not match [1,256,1024] fp32");
    }

    std::vector<float> output(kOutputElementCount, 0.0f);
    double latencyMs = 0.0;
    RunResult runResult = RunOfflineModel(caseName, modelBytes.bytes, inputBytes.bytes, output, latencyMs);
    if (!runResult.ok) {
        return runResult;
    }

    const std::vector<float> expected = BytesToFp32Vector(expectedBytes.bytes);
    MetricsResult metrics = CompareFp32Tensors(output, expected);

    RunResult result;
    result.ok = metrics.ok;
    result.caseName = caseName;
    result.deviceType = kDeviceType;
    result.errorStage = metrics.errorStage;
    result.errorMessage = metrics.errorMessage;
    result.latencyMs = latencyMs;
    result.outputElementCount = output.size();
    result.maxAbsDiff = metrics.maxAbsDiff;
    result.meanAbsDiff = metrics.meanAbsDiff;
    result.cosine = metrics.cosine;
    result.finite = metrics.finite;
    return result;
}

StabilityResult RunStability(napi_env env, napi_value resourceManager, const std::string& caseName, int repeatCount)
{
    if (repeatCount <= 0) {
        repeatCount = 20;
    }

    StabilityResult result;
    result.caseName = caseName;
    result.repeatCount = repeatCount;
    result.minLatencyMs = std::numeric_limits<double>::max();

    double latencySum = 0.0;
    for (int index = 0; index < repeatCount; ++index) {
        RunResult run = RunOnce(env, resourceManager, caseName);
        result.lastRun = run;
        if (!run.ok) {
            result.errorStage = run.errorStage;
            result.errorMessage = run.errorMessage;
            break;
        }
        result.successCount += 1;
        result.minLatencyMs = std::min(result.minLatencyMs, run.latencyMs);
        result.maxLatencyMs = std::max(result.maxLatencyMs, run.latencyMs);
        latencySum += run.latencyMs;
    }

    if (result.successCount > 0) {
        result.avgLatencyMs = latencySum / static_cast<double>(result.successCount);
    } else {
        result.minLatencyMs = 0.0;
    }

    result.ok = result.successCount == repeatCount;
    return result;
}

} // namespace internvl
