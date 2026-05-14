#include "napi/native_api.h"
#include "rawfile/raw_file_manager.h"

#include "validation/napi_result.h"
#include "validation/nn_runtime_validator.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

enum class AsyncOperation {
    LoadModel,
    UnloadModel,
    RunOfficialSmoke,
    RunOfficialClassification,
    RunOnce,
    RunStability,
};

struct AsyncContext {
    napi_deferred deferred = nullptr;
    napi_async_work work = nullptr;
    NativeResourceManager* resourceManager = nullptr;
    AsyncOperation operation = AsyncOperation::RunOnce;
    std::string caseName;
    int repeatCount = 20;
    internvl::ModelStatusResult modelResult;
    internvl::OfficialSmokeResult officialSmokeResult;
    internvl::OfficialClassificationResult officialClassificationResult;
    internvl::RunResult runResult;
    internvl::StabilityResult stabilityResult;
};

internvl::ModelStatusResult ModelArgError(const std::string& message)
{
    internvl::ModelStatusResult result;
    result.loaded = internvl::IsModelLoaded();
    result.errorStage = "napi_arg_error";
    result.errorMessage = message;
    return result;
}

internvl::RunResult RunArgError(const std::string& message)
{
    internvl::RunResult result;
    result.errorStage = "napi_arg_error";
    result.errorMessage = message;
    return result;
}

internvl::StabilityResult StabilityArgError(const std::string& message)
{
    internvl::StabilityResult result;
    result.errorStage = "napi_arg_error";
    result.errorMessage = message;
    return result;
}

internvl::OfficialSmokeResult OfficialSmokeArgError(const std::string& message)
{
    internvl::OfficialSmokeResult result;
    result.errorStage = "napi_arg_error";
    result.errorMessage = message;
    return result;
}

internvl::OfficialClassificationResult OfficialClassificationArgError(const std::string& message)
{
    internvl::OfficialClassificationResult result;
    result.errorStage = "napi_arg_error";
    result.errorMessage = message;
    return result;
}

bool ReadStringArg(napi_env env, napi_value value, std::string& result)
{
    napi_valuetype valueType = napi_undefined;
    if (napi_typeof(env, value, &valueType) != napi_ok || valueType != napi_string) {
        return false;
    }

    std::size_t length = 0;
    if (napi_get_value_string_utf8(env, value, nullptr, 0, &length) != napi_ok) {
        return false;
    }

    std::vector<char> buffer(length + 1, '\0');
    std::size_t copied = 0;
    if (napi_get_value_string_utf8(env, value, buffer.data(), buffer.size(), &copied) != napi_ok) {
        return false;
    }

    result.assign(buffer.data(), copied);
    return true;
}

bool ReadInt32Arg(napi_env env, napi_value value, std::int32_t& result)
{
    napi_valuetype valueType = napi_undefined;
    if (napi_typeof(env, value, &valueType) != napi_ok || valueType != napi_number) {
        return false;
    }
    return napi_get_value_int32(env, value, &result) == napi_ok;
}

bool IsObjectArg(napi_env env, napi_value value)
{
    napi_valuetype valueType = napi_undefined;
    return napi_typeof(env, value, &valueType) == napi_ok && valueType == napi_object;
}

napi_value ErrorResultToNapiValue(napi_env env, AsyncOperation operation, const std::string& message)
{
    switch (operation) {
        case AsyncOperation::LoadModel:
        case AsyncOperation::UnloadModel:
            return internvl::ToNapiValue(env, ModelArgError(message));
        case AsyncOperation::RunOfficialSmoke:
            return internvl::ToNapiValue(env, OfficialSmokeArgError(message));
        case AsyncOperation::RunOfficialClassification:
            return internvl::ToNapiValue(env, OfficialClassificationArgError(message));
        case AsyncOperation::RunStability:
            return internvl::ToNapiValue(env, StabilityArgError(message));
        case AsyncOperation::RunOnce:
        default:
            return internvl::ToNapiValue(env, RunArgError(message));
    }
}

napi_value CreateResolvedPromise(napi_env env, napi_value value)
{
    napi_value promise = nullptr;
    napi_deferred deferred = nullptr;
    napi_create_promise(env, &deferred, &promise);
    napi_resolve_deferred(env, deferred, value);
    return promise;
}

void ReleaseAsyncContext(AsyncContext* context)
{
    if (context == nullptr) {
        return;
    }
    if (context->resourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(context->resourceManager);
        context->resourceManager = nullptr;
    }
    delete context;
}

void ExecuteAsync(napi_env env, void* data)
{
    (void)env;
    auto* context = static_cast<AsyncContext*>(data);
    switch (context->operation) {
        case AsyncOperation::LoadModel:
            context->modelResult = internvl::LoadModel(context->resourceManager);
            break;
        case AsyncOperation::UnloadModel:
            context->modelResult = internvl::UnloadModel();
            break;
        case AsyncOperation::RunOfficialSmoke:
            context->officialSmokeResult = internvl::RunOfficialSmoke(context->resourceManager);
            break;
        case AsyncOperation::RunOfficialClassification:
            context->officialClassificationResult =
                internvl::RunOfficialClassification(context->resourceManager, context->caseName);
            break;
        case AsyncOperation::RunStability:
            context->stabilityResult = internvl::RunStability(
                context->resourceManager, context->caseName, context->repeatCount);
            break;
        case AsyncOperation::RunOnce:
        default:
            context->runResult = internvl::RunOnce(context->resourceManager, context->caseName);
            break;
    }
}

void CompleteAsync(napi_env env, napi_status status, void* data)
{
    auto* context = static_cast<AsyncContext*>(data);
    napi_value value = nullptr;
    if (status != napi_ok) {
        value = ErrorResultToNapiValue(env, context->operation, "Native async work failed");
    } else {
        switch (context->operation) {
            case AsyncOperation::LoadModel:
            case AsyncOperation::UnloadModel:
                value = internvl::ToNapiValue(env, context->modelResult);
                break;
            case AsyncOperation::RunOfficialSmoke:
                value = internvl::ToNapiValue(env, context->officialSmokeResult);
                break;
            case AsyncOperation::RunOfficialClassification:
                value = internvl::ToNapiValue(env, context->officialClassificationResult);
                break;
            case AsyncOperation::RunStability:
                value = internvl::ToNapiValue(env, context->stabilityResult);
                break;
            case AsyncOperation::RunOnce:
            default:
                value = internvl::ToNapiValue(env, context->runResult);
                break;
        }
    }

    napi_resolve_deferred(env, context->deferred, value);
    napi_delete_async_work(env, context->work);
    ReleaseAsyncContext(context);
}

napi_value QueueAsync(napi_env env, AsyncContext* context, const char* workName)
{
    napi_value promise = nullptr;
    if (napi_create_promise(env, &context->deferred, &promise) != napi_ok) {
        ReleaseAsyncContext(context);
        napi_value undefinedValue = nullptr;
        napi_get_undefined(env, &undefinedValue);
        return undefinedValue;
    }

    napi_value resourceName = nullptr;
    napi_create_string_utf8(env, workName, std::strlen(workName), &resourceName);
    napi_status status = napi_create_async_work(
        env, nullptr, resourceName, ExecuteAsync, CompleteAsync, context, &context->work);
    if (status != napi_ok) {
        napi_value value = ErrorResultToNapiValue(env, context->operation, "napi_create_async_work failed");
        napi_resolve_deferred(env, context->deferred, value);
        ReleaseAsyncContext(context);
        return promise;
    }

    status = napi_queue_async_work(env, context->work);
    if (status != napi_ok) {
        napi_value value = ErrorResultToNapiValue(env, context->operation, "napi_queue_async_work failed");
        napi_resolve_deferred(env, context->deferred, value);
        napi_delete_async_work(env, context->work);
        ReleaseAsyncContext(context);
        return promise;
    }

    return promise;
}

napi_value ListTestCases(napi_env env, napi_callback_info info)
{
    (void)info;
    return internvl::StringArrayToNapiValue(env, internvl::ListTestCaseNames());
}

napi_value LoadModel(napi_env env, napi_callback_info info)
{
    std::size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 1 || !IsObjectArg(env, args[0])) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(env, ModelArgError("loadModel requires resourceManager")));
    }

    NativeResourceManager* resourceManager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
    if (resourceManager == nullptr) {
        return CreateResolvedPromise(
            env, internvl::ToNapiValue(env, ModelArgError("Failed to initialize NativeResourceManager")));
    }

    auto* context = new AsyncContext();
    context->operation = AsyncOperation::LoadModel;
    context->resourceManager = resourceManager;
    return QueueAsync(env, context, "InternVLLoadModel");
}

napi_value UnloadModel(napi_env env, napi_callback_info info)
{
    (void)info;
    auto* context = new AsyncContext();
    context->operation = AsyncOperation::UnloadModel;
    return QueueAsync(env, context, "InternVLUnloadModel");
}

napi_value RunOfficialSmoke(napi_env env, napi_callback_info info)
{
    std::size_t argc = 1;
    napi_value args[1] = {nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 1 || !IsObjectArg(env, args[0])) {
        return CreateResolvedPromise(
            env, internvl::ToNapiValue(env, OfficialSmokeArgError("runOfficialSmoke requires resourceManager")));
    }

    NativeResourceManager* resourceManager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
    if (resourceManager == nullptr) {
        return CreateResolvedPromise(
            env, internvl::ToNapiValue(env, OfficialSmokeArgError("Failed to initialize NativeResourceManager")));
    }

    auto* context = new AsyncContext();
    context->operation = AsyncOperation::RunOfficialSmoke;
    context->resourceManager = resourceManager;
    return QueueAsync(env, context, "InternVLOfficialSmoke");
}

napi_value RunOfficialClassification(napi_env env, napi_callback_info info)
{
    std::size_t argc = 2;
    napi_value args[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 2 || !IsObjectArg(env, args[0])) {
        return CreateResolvedPromise(
            env, internvl::ToNapiValue(
                env, OfficialClassificationArgError("runOfficialClassification requires resourceManager and imageName")));
    }

    std::string imageName;
    if (!ReadStringArg(env, args[1], imageName)) {
        return CreateResolvedPromise(
            env, internvl::ToNapiValue(
                env, OfficialClassificationArgError("runOfficialClassification imageName must be a string")));
    }

    NativeResourceManager* resourceManager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
    if (resourceManager == nullptr) {
        return CreateResolvedPromise(
            env, internvl::ToNapiValue(
                env, OfficialClassificationArgError("Failed to initialize NativeResourceManager")));
    }

    auto* context = new AsyncContext();
    context->operation = AsyncOperation::RunOfficialClassification;
    context->resourceManager = resourceManager;
    context->caseName = imageName;
    return QueueAsync(env, context, "InternVLOfficialClassification");
}

napi_value RunOnce(napi_env env, napi_callback_info info)
{
    std::size_t argc = 2;
    napi_value args[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 2 || !IsObjectArg(env, args[0])) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, RunArgError("runOnce requires resourceManager and caseName")));
    }

    std::string caseName;
    if (!ReadStringArg(env, args[1], caseName)) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, RunArgError("runOnce caseName must be a string")));
    }

    NativeResourceManager* resourceManager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
    if (resourceManager == nullptr) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, RunArgError("Failed to initialize NativeResourceManager")));
    }

    auto* context = new AsyncContext();
    context->operation = AsyncOperation::RunOnce;
    context->resourceManager = resourceManager;
    context->caseName = caseName;
    return QueueAsync(env, context, "InternVLRunOnce");
}

napi_value RunStability(napi_env env, napi_callback_info info)
{
    std::size_t argc = 3;
    napi_value args[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 3 || !IsObjectArg(env, args[0])) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, StabilityArgError("runStability requires resourceManager, caseName, and repeatCount")));
    }

    std::string caseName;
    if (!ReadStringArg(env, args[1], caseName)) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, StabilityArgError("runStability caseName must be a string")));
    }

    std::int32_t repeatCount = 20;
    if (!ReadInt32Arg(env, args[2], repeatCount)) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, StabilityArgError("runStability repeatCount must be a number")));
    }

    NativeResourceManager* resourceManager = OH_ResourceManager_InitNativeResourceManager(env, args[0]);
    if (resourceManager == nullptr) {
        return CreateResolvedPromise(env, internvl::ToNapiValue(
            env, StabilityArgError("Failed to initialize NativeResourceManager")));
    }

    auto* context = new AsyncContext();
    context->operation = AsyncOperation::RunStability;
    context->resourceManager = resourceManager;
    context->caseName = caseName;
    context->repeatCount = repeatCount;
    return QueueAsync(env, context, "InternVLRunStability");
}

} // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        {"listTestCases", nullptr, ListTestCases, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"loadModel", nullptr, LoadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"unloadModel", nullptr, UnloadModel, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runOfficialSmoke", nullptr, RunOfficialSmoke, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runOfficialClassification", nullptr, RunOfficialClassification, nullptr, nullptr, nullptr, napi_default,
         nullptr},
        {"runOnce", nullptr, RunOnce, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runStability", nullptr, RunStability, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = { 0 },
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
}
