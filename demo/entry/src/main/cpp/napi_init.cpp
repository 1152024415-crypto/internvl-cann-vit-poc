#include "napi/native_api.h"

#include "validation/napi_result.h"
#include "validation/nn_runtime_validator.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

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

napi_value ListTestCases(napi_env env, napi_callback_info info)
{
    (void)info;
    return internvl::StringArrayToNapiValue(env, internvl::ListTestCaseNames());
}

napi_value RunOnce(napi_env env, napi_callback_info info)
{
    std::size_t argc = 2;
    napi_value args[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 2 || !IsObjectArg(env, args[0])) {
        return internvl::ToNapiValue(env, RunArgError("runOnce requires resourceManager and caseName"));
    }

    std::string caseName;
    if (!ReadStringArg(env, args[1], caseName)) {
        return internvl::ToNapiValue(env, RunArgError("runOnce caseName must be a string"));
    }

    return internvl::ToNapiValue(env, internvl::RunOnce(env, args[0], caseName));
}

napi_value RunStability(napi_env env, napi_callback_info info)
{
    std::size_t argc = 3;
    napi_value args[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);

    if (argc < 3 || !IsObjectArg(env, args[0])) {
        return internvl::ToNapiValue(env,
                                     StabilityArgError("runStability requires resourceManager, caseName, and repeatCount"));
    }

    std::string caseName;
    if (!ReadStringArg(env, args[1], caseName)) {
        return internvl::ToNapiValue(env, StabilityArgError("runStability caseName must be a string"));
    }

    std::int32_t repeatCount = 20;
    if (!ReadInt32Arg(env, args[2], repeatCount)) {
        return internvl::ToNapiValue(env, StabilityArgError("runStability repeatCount must be a number"));
    }

    return internvl::ToNapiValue(env, internvl::RunStability(env, args[0], caseName, repeatCount));
}

} // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        {"listTestCases", nullptr, ListTestCases, nullptr, nullptr, nullptr, napi_default, nullptr},
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
