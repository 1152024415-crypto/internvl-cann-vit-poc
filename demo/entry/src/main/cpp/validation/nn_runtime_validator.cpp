#include "nn_runtime_validator.h"

#include "rawfile_loader.h"
#include "tensor_metrics.h"
#include "validation_constants.h"

#include "hilog/log.h"
#include "neural_network_runtime/neural_network_core.h"

#if defined(INTERNVL_HAS_HIAI_FOUNDATION) && __has_include("CANNKit/hiai_options.h") && \
    __has_include("CANNKit/hiai_helper.h")
#define INTERNVL_ENABLE_HIAI_OPTIONS 1
#include "CANNKit/hiai_helper.h"
#include "CANNKit/hiai_options.h"
#else
#define INTERNVL_ENABLE_HIAI_OPTIONS 0
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>

namespace internvl {

namespace {

constexpr unsigned int kLogDomain = 0x0000;
constexpr const char* kLogTag = "InternVLNative";
constexpr const char* kRequiredNpuName = "HIAI_F";
constexpr const char* kOfficialSampleModelFile = "official_squeezenet_hiai.om";

std::mutex gRuntimeMutex;
OH_NNExecutor* gExecutor = nullptr;
size_t gDeviceId = 0;
std::string gDeviceLabel;

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

void LogStage(const char* stage)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, kLogDomain, kLogTag, "stage=%{public}s", stage);
}

void LogStageError(const char* stage, const std::string& message)
{
    OH_LOG_Print(LOG_APP, LOG_ERROR, kLogDomain, kLogTag, "stage=%{public}s error=%{public}s", stage,
                 message.c_str());
}

void LogStageInfo(const char* stage, const std::string& message)
{
    OH_LOG_Print(LOG_APP, LOG_INFO, kLogDomain, kLogTag, "stage=%{public}s %{public}s", stage, message.c_str());
}

int ReturnCodeToInt(OH_NN_ReturnCode code)
{
    return static_cast<int>(code);
}

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

DeviceSelection SelectHiaiFDevice()
{
    const size_t* deviceIds = nullptr;
    uint32_t deviceCount = 0;
    DeviceSelection selection;

    const OH_NN_ReturnCode code = OH_NNDevice_GetAllDevicesID(&deviceIds, &deviceCount);
    if (code != OH_NN_SUCCESS || deviceIds == nullptr || deviceCount == 0) {
        selection.errorMessage = "OH_NNDevice_GetAllDevicesID failed";
        return selection;
    }

    std::ostringstream available;
    for (uint32_t index = 0; index < deviceCount; ++index) {
        if (index > 0) {
            available << "; ";
        }
        available << MakeDeviceLabel(deviceIds[index]);

        const char* deviceName = nullptr;
        if (OH_NNDevice_GetName(deviceIds[index], &deviceName) == OH_NN_SUCCESS && deviceName != nullptr &&
            std::string(deviceName) == kRequiredNpuName) {
            selection.ok = true;
            selection.id = deviceIds[index];
            selection.label = MakeDeviceLabel(selection.id);
            return selection;
        }
    }

    selection.errorMessage = std::string(kRequiredNpuName) + " device not found. Available devices: " +
                             available.str();
    return selection;
}

void LogHiaiCompatibility(const std::vector<std::uint8_t>& modelBytes)
{
#if INTERNVL_ENABLE_HIAI_OPTIONS
    const HiAI_Compatibility compatibility =
        HMS_HiAICompatibility_CheckFromBuffer(modelBytes.data(), modelBytes.size());
    LogStageInfo("hiai_compatibility_check",
                 "compatibility=" + std::to_string(static_cast<int>(compatibility)));
#else
    LogStageInfo("hiai_compatibility_check", "skipped=missing_hiai_foundation_headers_or_library");
#endif
}

OH_NN_ReturnCode SetOfficialHiaiBuildOptions(OH_NNCompilation* compilation)
{
#if INTERNVL_ENABLE_HIAI_OPTIONS
    OH_NN_ReturnCode code = HMS_HiAIOptions_SetBandMode(compilation, HiAI_BandMode::HIAI_BANDMODE_NORMAL);
    if (code != OH_NN_SUCCESS) {
        LogStageError("hiai_set_band_mode_failed",
                      "HMS_HiAIOptions_SetBandMode failed code=" + std::to_string(ReturnCodeToInt(code)));
        return code;
    }

    std::vector<HiAI_ExecuteDevice> executeDevices {HiAI_ExecuteDevice::HIAI_EXECUTE_DEVICE_NPU};
    code = HMS_HiAIOptions_SetModelDeviceOrder(compilation, executeDevices.data(), executeDevices.size());
    if (code != OH_NN_SUCCESS) {
        LogStageError("hiai_set_model_device_order_failed",
                      "HMS_HiAIOptions_SetModelDeviceOrder failed code=" + std::to_string(ReturnCodeToInt(code)));
        return code;
    }

    LogStageInfo("hiai_build_options_done", "band_mode=normal execute_device_order=npu");
    return OH_NN_SUCCESS;
#else
    (void)compilation;
    LogStageInfo("hiai_build_options_done", "skipped=missing_hiai_foundation_headers_or_library");
    return OH_NN_SUCCESS;
#endif
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

ModelStatusResult ModelFail(const std::string& stage,
                            const std::string& message,
                            const std::chrono::steady_clock::time_point& start)
{
    const auto end = std::chrono::steady_clock::now();
    ModelStatusResult result;
    result.loaded = gExecutor != nullptr;
    result.deviceType = gDeviceLabel;
    result.errorStage = stage;
    result.errorMessage = message;
    result.latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    LogStageError(stage.c_str(), message);
    return result;
}

OfficialSmokeResult OfficialSmokeFail(const std::string& stage,
                                      const std::string& message,
                                      const std::chrono::steady_clock::time_point& start,
                                      const std::string& deviceLabel,
                                      std::size_t modelByteCount)
{
    const auto end = std::chrono::steady_clock::now();
    OfficialSmokeResult result;
    result.deviceType = deviceLabel;
    result.errorStage = stage;
    result.errorMessage = message;
    result.modelByteCount = modelByteCount;
    result.latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    LogStageError(stage.c_str(), message);
    return result;
}

void DestroyExecutor(OH_NNExecutor** executor)
{
    if (executor != nullptr && *executor != nullptr) {
        OH_NNExecutor_Destroy(executor);
        *executor = nullptr;
    }
}

void DestroyCompilation(OH_NNCompilation** compilation)
{
    if (compilation != nullptr && *compilation != nullptr) {
        OH_NNCompilation_Destroy(compilation);
        *compilation = nullptr;
    }
}

void DestroyTensor(NN_Tensor** tensor)
{
    if (tensor != nullptr && *tensor != nullptr) {
        OH_NNTensor_Destroy(tensor);
        *tensor = nullptr;
    }
}

void DestroyTensorDesc(NN_TensorDesc** tensorDesc)
{
    if (tensorDesc != nullptr && *tensorDesc != nullptr) {
        OH_NNTensorDesc_Destroy(tensorDesc);
        *tensorDesc = nullptr;
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

RunResult ValidateExecutorIO(const std::string& caseName, OH_NNExecutor* executor)
{
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NN_ReturnCode code = OH_NNExecutor_GetInputCount(executor, &inputCount);
    if (code != OH_NN_SUCCESS || inputCount != 1) {
        return Fail(caseName, "input_size_mismatch", "Expected exactly one model input");
    }
    code = OH_NNExecutor_GetOutputCount(executor, &outputCount);
    if (code != OH_NN_SUCCESS || outputCount != 1) {
        return Fail(caseName, "output_size_mismatch", "Expected exactly one model output");
    }

    RunResult result;
    result.ok = true;
    result.caseName = caseName;
    return result;
}

RunResult RunLoadedModel(const std::string& caseName,
                         const std::vector<std::uint8_t>& inputBytes,
                         std::vector<float>& output,
                         double& latencyMs)
{
    RunResult ioCheck = ValidateExecutorIO(caseName, gExecutor);
    if (!ioCheck.ok) {
        return ioCheck;
    }

    NN_TensorDesc* inputDesc = nullptr;
    NN_TensorDesc* outputDesc = nullptr;
    NN_Tensor* inputTensor = nullptr;
    NN_Tensor* outputTensor = nullptr;

    const auto start = std::chrono::steady_clock::now();
    inputDesc = OH_NNExecutor_CreateInputTensorDesc(gExecutor, 0);
    outputDesc = OH_NNExecutor_CreateOutputTensorDesc(gExecutor, 0);
    if (inputDesc == nullptr || outputDesc == nullptr) {
        DestroyTensorDesc(&outputDesc);
        DestroyTensorDesc(&inputDesc);
        return Fail(caseName, "executor_create_failed", "Failed to create input or output tensor descriptors");
    }

    inputTensor = OH_NNTensor_CreateWithSize(gDeviceId, inputDesc, kInputByteCount);
    outputTensor = OH_NNTensor_CreateWithSize(gDeviceId, outputDesc, kOutputByteCount);
    DestroyTensorDesc(&outputDesc);
    DestroyTensorDesc(&inputDesc);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        DestroyTensor(&outputTensor);
        DestroyTensor(&inputTensor);
        return Fail(caseName, "executor_create_failed", "Failed to create input or output tensors");
    }

    void* inputBuffer = OH_NNTensor_GetDataBuffer(inputTensor);
    void* outputBuffer = OH_NNTensor_GetDataBuffer(outputTensor);
    if (inputBuffer == nullptr || outputBuffer == nullptr) {
        DestroyTensor(&outputTensor);
        DestroyTensor(&inputTensor);
        return Fail(caseName, "executor_create_failed", "Failed to get tensor data buffers");
    }

    std::memcpy(inputBuffer, inputBytes.data(), kInputByteCount);
    NN_Tensor* inputTensors[] = {inputTensor};
    NN_Tensor* outputTensors[] = {outputTensor};
    const OH_NN_ReturnCode code = OH_NNExecutor_RunSync(gExecutor, inputTensors, 1, outputTensors, 1);

    if (code == OH_NN_SUCCESS) {
        std::memcpy(output.data(), outputBuffer, kOutputByteCount);
    }
    OutputShapeCheck outputShape = code == OH_NN_SUCCESS ? CheckOutputShape(gExecutor) : OutputShapeCheck{};

    DestroyTensor(&outputTensor);
    DestroyTensor(&inputTensor);

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
    result.deviceType = gDeviceLabel;
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

ModelStatusResult LoadModel(NativeResourceManager* resourceManager)
{
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto start = std::chrono::steady_clock::now();

    if (gExecutor != nullptr) {
        ModelStatusResult result;
        result.ok = true;
        result.loaded = true;
        result.deviceType = gDeviceLabel;
        return result;
    }

    LogStage("load_model_read_om_start");
    auto modelBytes = ReadRawfile(resourceManager, kModelFile);
    if (!modelBytes.ok) {
        return ModelFail(modelBytes.errorStage, modelBytes.errorMessage, start);
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, kLogDomain, kLogTag, "stage=load_model_read_om_done om_bytes=%{public}zu",
                 modelBytes.bytes.size());

    LogStage("load_model_select_hiai_f");
    DeviceSelection device = SelectHiaiFDevice();
    if (!device.ok) {
        return ModelFail("device_selection_failed", device.errorMessage, start);
    }
    OH_LOG_Print(LOG_APP, LOG_INFO, kLogDomain, kLogTag, "stage=load_model_select_hiai_f selected_device=%{public}s",
                 device.label.c_str());

    OH_NNCompilation* compilation = nullptr;
    OH_NNExecutor* executor = nullptr;

    LogStage("load_model_construct_compilation");
    LogHiaiCompatibility(modelBytes.bytes);
    compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelBytes.bytes.data(), modelBytes.bytes.size());
    if (compilation == nullptr) {
        return ModelFail("om_compilation_failed", "OH_NNCompilation_ConstructWithOfflineModelBuffer failed", start);
    }

    OH_NN_ReturnCode code = OH_NNCompilation_SetDevice(compilation, device.id);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return ModelFail("device_selection_failed",
                         "OH_NNCompilation_SetDevice failed code=" + std::to_string(ReturnCodeToInt(code)) +
                             " device=" + device.label,
                         start);
    }

    LogStage("load_model_set_hiai_build_options");
    code = SetOfficialHiaiBuildOptions(compilation);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return ModelFail("hiai_build_options_failed",
                         "Official HiAI build option setup failed code=" + std::to_string(ReturnCodeToInt(code)),
                         start);
    }

    LogStage("load_model_build_start");
    code = OH_NNCompilation_Build(compilation);
    if (code != OH_NN_SUCCESS) {
        DestroyCompilation(&compilation);
        return ModelFail("om_compilation_failed",
                         "OH_NNCompilation_Build failed code=" + std::to_string(ReturnCodeToInt(code)),
                         start);
    }

    LogStage("load_model_create_executor");
    executor = OH_NNExecutor_Construct(compilation);
    if (executor == nullptr) {
        DestroyCompilation(&compilation);
        return ModelFail("executor_create_failed", "OH_NNExecutor_Construct failed", start);
    }

    RunResult ioCheck = ValidateExecutorIO("model", executor);
    if (!ioCheck.ok) {
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
        return ModelFail(ioCheck.errorStage, ioCheck.errorMessage, start);
    }

    DestroyCompilation(&compilation);
    gExecutor = executor;
    gDeviceId = device.id;
    gDeviceLabel = device.label;

    const auto end = std::chrono::steady_clock::now();
    ModelStatusResult result;
    result.ok = true;
    result.loaded = true;
    result.deviceType = gDeviceLabel;
    result.latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    LogStage("load_model_done");
    return result;
}

ModelStatusResult LoadModel(napi_env env, napi_value resourceManager)
{
    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    ModelStatusResult result = LoadModel(nativeResourceManager);
    if (nativeResourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
    }
    return result;
}

ModelStatusResult UnloadModel()
{
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    DestroyExecutor(&gExecutor);
    gDeviceId = 0;
    gDeviceLabel.clear();

    ModelStatusResult result;
    result.ok = true;
    result.loaded = false;
    LogStage("unload_model_done");
    return result;
}

bool IsModelLoaded()
{
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    return gExecutor != nullptr;
}

OfficialSmokeResult RunOfficialSmoke(NativeResourceManager* resourceManager)
{
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    const auto start = std::chrono::steady_clock::now();
    std::string deviceLabel;
    std::size_t modelByteCount = 0;

    OH_NNCompilation* compilation = nullptr;
    OH_NNExecutor* executor = nullptr;
    NN_TensorDesc* inputDesc = nullptr;
    NN_TensorDesc* outputDesc = nullptr;
    NN_Tensor* inputTensor = nullptr;
    NN_Tensor* outputTensor = nullptr;

    auto cleanup = [&]() {
        DestroyTensor(&outputTensor);
        DestroyTensor(&inputTensor);
        DestroyTensorDesc(&outputDesc);
        DestroyTensorDesc(&inputDesc);
        DestroyExecutor(&executor);
        DestroyCompilation(&compilation);
    };
    auto fail = [&](const std::string& stage, const std::string& message) {
        cleanup();
        return OfficialSmokeFail(stage, message, start, deviceLabel, modelByteCount);
    };

    LogStage("official_smoke_read_om_start");
    auto modelBytes = ReadRawfile(resourceManager, kOfficialSampleModelFile);
    if (!modelBytes.ok) {
        return fail(modelBytes.errorStage, modelBytes.errorMessage);
    }
    modelByteCount = modelBytes.bytes.size();
    OH_LOG_Print(LOG_APP, LOG_INFO, kLogDomain, kLogTag,
                 "stage=official_smoke_read_om_done om_bytes=%{public}zu", modelByteCount);

    LogStage("official_smoke_select_hiai_f");
    DeviceSelection device = SelectHiaiFDevice();
    if (!device.ok) {
        return fail("device_selection_failed", device.errorMessage);
    }
    deviceLabel = device.label;
    OH_LOG_Print(LOG_APP, LOG_INFO, kLogDomain, kLogTag,
                 "stage=official_smoke_select_hiai_f selected_device=%{public}s", deviceLabel.c_str());

    LogStage("official_smoke_construct_compilation");
    LogHiaiCompatibility(modelBytes.bytes);
    compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelBytes.bytes.data(), modelBytes.bytes.size());
    if (compilation == nullptr) {
        return fail("official_smoke_compilation_failed",
                    "OH_NNCompilation_ConstructWithOfflineModelBuffer failed");
    }

    OH_NN_ReturnCode code = OH_NNCompilation_SetDevice(compilation, device.id);
    if (code != OH_NN_SUCCESS) {
        return fail("device_selection_failed",
                    "OH_NNCompilation_SetDevice failed code=" + std::to_string(ReturnCodeToInt(code)) +
                        " device=" + deviceLabel);
    }

    LogStage("official_smoke_set_hiai_build_options");
    code = SetOfficialHiaiBuildOptions(compilation);
    if (code != OH_NN_SUCCESS) {
        return fail("hiai_build_options_failed",
                    "Official HiAI build option setup failed code=" + std::to_string(ReturnCodeToInt(code)));
    }

    LogStage("official_smoke_build_start");
    code = OH_NNCompilation_Build(compilation);
    if (code != OH_NN_SUCCESS) {
        return fail("official_smoke_compilation_failed",
                    "OH_NNCompilation_Build failed code=" + std::to_string(ReturnCodeToInt(code)));
    }

    LogStage("official_smoke_create_executor");
    executor = OH_NNExecutor_Construct(compilation);
    if (executor == nullptr) {
        return fail("official_smoke_executor_failed", "OH_NNExecutor_Construct failed");
    }

    size_t inputCount = 0;
    size_t outputCount = 0;
    code = OH_NNExecutor_GetInputCount(executor, &inputCount);
    if (code != OH_NN_SUCCESS || inputCount != 1) {
        return fail("official_smoke_io_mismatch", "Expected exactly one official sample input");
    }
    code = OH_NNExecutor_GetOutputCount(executor, &outputCount);
    if (code != OH_NN_SUCCESS || outputCount != 1) {
        return fail("official_smoke_io_mismatch", "Expected exactly one official sample output");
    }

    inputDesc = OH_NNExecutor_CreateInputTensorDesc(executor, 0);
    outputDesc = OH_NNExecutor_CreateOutputTensorDesc(executor, 0);
    if (inputDesc == nullptr || outputDesc == nullptr) {
        return fail("official_smoke_tensor_failed", "Failed to create official sample tensor descriptors");
    }

    inputTensor = OH_NNTensor_Create(device.id, inputDesc);
    outputTensor = OH_NNTensor_Create(device.id, outputDesc);
    DestroyTensorDesc(&outputDesc);
    DestroyTensorDesc(&inputDesc);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        return fail("official_smoke_tensor_failed", "Failed to create official sample tensors");
    }

    void* inputBuffer = OH_NNTensor_GetDataBuffer(inputTensor);
    std::size_t inputByteCount = 0;
    code = OH_NNTensor_GetSize(inputTensor, &inputByteCount);
    if (code != OH_NN_SUCCESS || inputBuffer == nullptr || inputByteCount == 0) {
        return fail("official_smoke_tensor_failed", "Failed to get official sample input buffer");
    }
    std::memset(inputBuffer, 0, inputByteCount);

    NN_Tensor* inputTensors[] = {inputTensor};
    NN_Tensor* outputTensors[] = {outputTensor};
    LogStage("official_smoke_run_sync_start");
    code = OH_NNExecutor_RunSync(executor, inputTensors, 1, outputTensors, 1);
    if (code != OH_NN_SUCCESS) {
        return fail("official_smoke_run_sync_failed",
                    "OH_NNExecutor_RunSync failed code=" + std::to_string(ReturnCodeToInt(code)));
    }

    void* outputBuffer = OH_NNTensor_GetDataBuffer(outputTensor);
    std::size_t outputByteCount = 0;
    code = OH_NNTensor_GetSize(outputTensor, &outputByteCount);
    if (code != OH_NN_SUCCESS || outputBuffer == nullptr || outputByteCount == 0) {
        return fail("official_smoke_tensor_failed", "Failed to get official sample output buffer");
    }

    cleanup();

    const auto end = std::chrono::steady_clock::now();
    OfficialSmokeResult result;
    result.ok = true;
    result.deviceType = deviceLabel;
    result.latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
    result.modelByteCount = modelByteCount;
    result.inputCount = inputCount;
    result.outputCount = outputCount;
    result.inputByteCount = inputByteCount;
    result.outputByteCount = outputByteCount;
    LogStage("official_smoke_done");
    return result;
}

OfficialSmokeResult RunOfficialSmoke(napi_env env, napi_value resourceManager)
{
    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    OfficialSmokeResult result = RunOfficialSmoke(nativeResourceManager);
    if (nativeResourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
    }
    return result;
}

RunResult RunOnce(NativeResourceManager* resourceManager, const std::string& caseName)
{
    std::lock_guard<std::mutex> lock(gRuntimeMutex);
    if (gExecutor == nullptr) {
        return Fail(caseName, "model_not_loaded", "Load the OM model before running validation");
    }

    const TestCase* testCase = FindCase(caseName);
    if (testCase == nullptr) {
        return Fail(caseName, "missing_artifact", "Unknown validation test case");
    }

    auto inputBytes = ReadRawfile(resourceManager, testCase->inputFile);
    if (!inputBytes.ok) {
        return Fail(caseName, inputBytes.errorStage, inputBytes.errorMessage);
    }
    if (inputBytes.bytes.size() != kInputByteCount) {
        return Fail(caseName, "input_size_mismatch", "Input rawfile byte count does not match [1,3,448,448] fp32");
    }

    auto expectedBytes = ReadRawfile(resourceManager, testCase->expectedOutputFile);
    if (!expectedBytes.ok) {
        return Fail(caseName, expectedBytes.errorStage, expectedBytes.errorMessage);
    }
    if (expectedBytes.bytes.size() != kOutputByteCount) {
        return Fail(caseName, "output_size_mismatch",
                    "Expected output rawfile byte count does not match [1,256,1024] fp32");
    }
    auto metadataBytes = ReadRawfile(resourceManager, testCase->metadataFile);
    if (!metadataBytes.ok) {
        return Fail(caseName, metadataBytes.errorStage, metadataBytes.errorMessage);
    }
    MetricsResult metadataResult = ValidateMetadata(metadataBytes.bytes, *testCase);
    if (!metadataResult.ok) {
        return Fail(caseName, metadataResult.errorStage, metadataResult.errorMessage);
    }

    std::vector<float> output(kOutputElementCount, 0.0f);
    double latencyMs = 0.0;
    RunResult runResult = RunLoadedModel(caseName, inputBytes.bytes, output, latencyMs);
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

RunResult RunOnce(napi_env env, napi_value resourceManager, const std::string& caseName)
{
    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    RunResult result = RunOnce(nativeResourceManager, caseName);
    if (nativeResourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
    }
    return result;
}

StabilityResult RunStability(NativeResourceManager* resourceManager, const std::string& caseName, int repeatCount)
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
        RunResult run = RunOnce(resourceManager, caseName);
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

StabilityResult RunStability(napi_env env, napi_value resourceManager, const std::string& caseName, int repeatCount)
{
    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    StabilityResult result = RunStability(nativeResourceManager, caseName, repeatCount);
    if (nativeResourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
    }
    return result;
}

} // namespace internvl
