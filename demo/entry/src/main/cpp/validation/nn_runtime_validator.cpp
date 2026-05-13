#include "nn_runtime_validator.h"

#include "rawfile_loader.h"
#include "tensor_metrics.h"
#include "validation_constants.h"

#include "neural_network_runtime/neural_network_core.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>

namespace internvl {

namespace {

struct DeviceSelection {
    bool ok = false;
    size_t id = 0;
    std::string label;
    std::string errorMessage;
};

struct OutputShapeCheck {
    bool ok = false;
    std::string shape;
};

const char* DeviceTypeName(OH_NN_DeviceType type)
{
    switch (type) {
        case OH_NN_CPU:
            return "CPU";
        case OH_NN_GPU:
            return "GPU";
        case OH_NN_ACCELERATOR:
            return "ACCELERATOR";
        case OH_NN_OTHERS:
        default:
            return "OTHERS";
    }
}

std::string MakeDeviceLabel(size_t deviceId)
{
    const char* deviceName = nullptr;
    OH_NN_DeviceType deviceType = OH_NN_OTHERS;
    const OH_NN_ReturnCode nameCode = OH_NNDevice_GetName(deviceId, &deviceName);
    const OH_NN_ReturnCode typeCode = OH_NNDevice_GetType(deviceId, &deviceType);

    std::ostringstream label;
    label << "id=" << deviceId;
    if (nameCode == OH_NN_SUCCESS && deviceName != nullptr) {
        label << " name=" << deviceName;
    }
    if (typeCode == OH_NN_SUCCESS) {
        label << " type=" << DeviceTypeName(deviceType);
    }
    return label.str();
}

DeviceSelection SelectDevice()
{
    const size_t* deviceIds = nullptr;
    uint32_t deviceCount = 0;
    DeviceSelection selection;

    const OH_NN_ReturnCode code = OH_NNDevice_GetAllDevicesID(&deviceIds, &deviceCount);
    if (code != OH_NN_SUCCESS || deviceIds == nullptr || deviceCount == 0) {
        selection.errorMessage = "OH_NNDevice_GetAllDevicesID failed";
        return selection;
    }

    for (uint32_t index = 0; index < deviceCount; ++index) {
        OH_NN_DeviceType deviceType = OH_NN_OTHERS;
        if (OH_NNDevice_GetType(deviceIds[index], &deviceType) == OH_NN_SUCCESS &&
            deviceType == OH_NN_ACCELERATOR) {
            selection.ok = true;
            selection.id = deviceIds[index];
            selection.label = MakeDeviceLabel(selection.id);
            return selection;
        }
    }

    selection.ok = true;
    selection.id = deviceIds[0];
    selection.label = MakeDeviceLabel(selection.id);
    return selection;
}

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

void DestroyTensor(NN_Tensor** tensor)
{
    if (tensor != nullptr && *tensor != nullptr) {
        OH_NNTensor_Destroy(tensor);
    }
}

void DestroyTensorDesc(NN_TensorDesc** tensorDesc)
{
    if (tensorDesc != nullptr && *tensorDesc != nullptr) {
        OH_NNTensorDesc_Destroy(tensorDesc);
    }
}

std::string ShapeToString(const int32_t* shape, uint32_t shapeLength)
{
    if (shape == nullptr) {
        return "unavailable";
    }

    std::ostringstream stream;
    stream << "[";
    for (uint32_t index = 0; index < shapeLength; ++index) {
        if (index > 0) {
            stream << ",";
        }
        stream << shape[index];
    }
    stream << "]";
    return stream.str();
}

OutputShapeCheck CheckOutputShape(OH_NNExecutor* executor)
{
    int32_t* outputShape = nullptr;
    uint32_t shapeLength = 0;
    const OH_NN_ReturnCode code = OH_NNExecutor_GetOutputShape(executor, 0, &outputShape, &shapeLength);
    OutputShapeCheck result;
    if (code != OH_NN_SUCCESS || outputShape == nullptr || shapeLength != 3) {
        result.shape = ShapeToString(outputShape, shapeLength);
        return result;
    }
    result.shape = ShapeToString(outputShape, shapeLength);
    result.ok = outputShape[0] == static_cast<int32_t>(kOutputN) &&
                outputShape[1] == static_cast<int32_t>(kOutputTokenCount) &&
                outputShape[2] == static_cast<int32_t>(kOutputHiddenSize);
    return result;
}

bool ContainsText(const std::string& text, const std::string& expected)
{
    return text.find(expected) != std::string::npos;
}

MetricsResult ValidateMetadata(const std::vector<std::uint8_t>& metadataBytes, const TestCase& testCase)
{
    MetricsResult result;
    const std::string metadata(metadataBytes.begin(), metadataBytes.end());
    const std::string caseFragment = "\"case_name\": \"" + std::string(testCase.name) + "\"";

    const bool valid = ContainsText(metadata, caseFragment) &&
                       ContainsText(metadata, kModelFile) &&
                       ContainsText(metadata, testCase.inputFile) &&
                       ContainsText(metadata, testCase.expectedOutputFile) &&
                       ContainsText(metadata, "\"byte_count\": 2408448") &&
                       ContainsText(metadata, "\"byte_count\": 1048576") &&
                       ContainsText(metadata, "\"cosine_min\": 0.999") &&
                       ContainsText(metadata, "\"mean_abs_diff_max\": 0.01");
    if (valid) {
        result.ok = true;
        result.finite = true;
        return result;
    }

    result.errorStage = "metadata_mismatch";
    result.errorMessage = "Validation metadata content does not match expected case artifacts";
    return result;
}

RunResult RunOfflineModel(const std::string& caseName,
                          const std::vector<std::uint8_t>& modelBytes,
                          const std::vector<std::uint8_t>& inputBytes,
                          std::vector<float>& output,
                          double& latencyMs)
{
    DeviceSelection device = SelectDevice();
    if (!device.ok) {
        return Fail(caseName, "device_selection_failed", device.errorMessage);
    }

    OH_NNCompilation* compilation = nullptr;
    OH_NNExecutor* executor = nullptr;
    NN_TensorDesc* inputDesc = nullptr;
    NN_TensorDesc* outputDesc = nullptr;
    NN_Tensor* inputTensor = nullptr;
    NN_Tensor* outputTensor = nullptr;

    // Keep all HarmonyOS NN calls in this file so SDK compatibility fixes stay isolated.
    const auto start = std::chrono::steady_clock::now();
    compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelBytes.data(), modelBytes.size());
    if (compilation == nullptr) {
        return Fail(caseName, "om_compilation_failed", "OH_NNCompilation_ConstructWithOfflineModelBuffer failed");
    }

    OH_NN_ReturnCode code = OH_NNCompilation_SetDevice(compilation, device.id);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return Fail(caseName, "device_selection_failed", "OH_NNCompilation_SetDevice failed: " + device.label);
    }

    code = OH_NNCompilation_Build(compilation);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return Fail(caseName, "om_compilation_failed", "OH_NNCompilation_Build failed");
    }

    executor = OH_NNExecutor_Construct(compilation);
    if (executor == nullptr) {
        DestroyCompilation(&compilation);
        return Fail(caseName, "executor_create_failed", "OH_NNExecutor_Construct failed");
    }

    size_t inputCount = 0;
    size_t outputCount = 0;
    code = OH_NNExecutor_GetInputCount(executor, &inputCount);
    if (code != OH_NN_SUCCESS || inputCount != 1) {
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "input_size_mismatch", "Expected exactly one model input");
    }
    code = OH_NNExecutor_GetOutputCount(executor, &outputCount);
    if (code != OH_NN_SUCCESS || outputCount != 1) {
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "output_size_mismatch", "Expected exactly one model output");
    }

    inputDesc = OH_NNExecutor_CreateInputTensorDesc(executor, 0);
    outputDesc = OH_NNExecutor_CreateOutputTensorDesc(executor, 0);
    if (inputDesc == nullptr || outputDesc == nullptr) {
        DestroyTensorDesc(&outputDesc);
        DestroyTensorDesc(&inputDesc);
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "executor_create_failed", "Failed to create input or output tensor descriptors");
    }

    inputTensor = OH_NNTensor_CreateWithSize(device.id, inputDesc, kInputByteCount);
    outputTensor = OH_NNTensor_CreateWithSize(device.id, outputDesc, kOutputByteCount);
    DestroyTensorDesc(&outputDesc);
    DestroyTensorDesc(&inputDesc);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        DestroyTensor(&outputTensor);
        DestroyTensor(&inputTensor);
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "executor_create_failed", "Failed to create input or output tensors");
    }

    void* inputBuffer = OH_NNTensor_GetDataBuffer(inputTensor);
    void* outputBuffer = OH_NNTensor_GetDataBuffer(outputTensor);
    if (inputBuffer == nullptr || outputBuffer == nullptr) {
        DestroyTensor(&outputTensor);
        DestroyTensor(&inputTensor);
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return Fail(caseName, "executor_create_failed", "Failed to get tensor data buffers");
    }

    std::memcpy(inputBuffer, inputBytes.data(), kInputByteCount);
    NN_Tensor* inputTensors[] = {inputTensor};
    NN_Tensor* outputTensors[] = {outputTensor};
    code = OH_NNExecutor_RunSync(executor, inputTensors, 1, outputTensors, 1);

    if (code == OH_NN_SUCCESS) {
        std::memcpy(output.data(), outputBuffer, kOutputByteCount);
    }
    OutputShapeCheck outputShape = code == OH_NN_SUCCESS ? CheckOutputShape(executor) : OutputShapeCheck{};

    DestroyTensor(&outputTensor);
    DestroyTensor(&inputTensor);
    DestroyExecutor(&executor);
    DestroyCompilation(&compilation);

    const auto end = std::chrono::steady_clock::now();
    latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    if (code != OH_NN_SUCCESS) {
        return Fail(caseName, "run_sync_failed", "OH_NNExecutor_RunSync failed");
    }
    if (!outputShape.ok) {
        RunResult result = Fail(caseName, "output_size_mismatch", "Runtime output shape is not [1,256,1024]");
        result.outputShape = outputShape.shape;
        return result;
    }

    RunResult result;
    result.ok = true;
    result.caseName = caseName;
    result.deviceType = device.label;
    result.latencyMs = latencyMs;
    result.outputElementCount = output.size();
    result.outputShapeOk = true;
    result.outputShape = outputShape.shape;
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
    auto metadataBytes = ReadRawfile(env, resourceManager, testCase->metadataFile);
    if (!metadataBytes.ok) {
        return Fail(caseName, metadataBytes.errorStage, metadataBytes.errorMessage);
    }
    MetricsResult metadataResult = ValidateMetadata(metadataBytes.bytes, *testCase);
    if (!metadataResult.ok) {
        RunResult result = Fail(caseName, metadataResult.errorStage, metadataResult.errorMessage);
        return result;
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
    result.deviceType = runResult.deviceType;
    result.errorStage = metrics.errorStage;
    result.errorMessage = metrics.errorMessage;
    result.latencyMs = latencyMs;
    result.outputElementCount = output.size();
    result.maxAbsDiff = metrics.maxAbsDiff;
    result.meanAbsDiff = metrics.meanAbsDiff;
    result.cosine = metrics.cosine;
    result.finite = metrics.finite;
    result.outputShapeOk = runResult.outputShapeOk;
    result.outputShape = runResult.outputShape;
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
