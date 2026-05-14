#pragma once

#include "napi/native_api.h"
#include "nn_runtime_validator.h"

#include <string>
#include <vector>

namespace internvl {

napi_value ToNapiValue(napi_env env, const RunResult& result);
napi_value ToNapiValue(napi_env env, const StabilityResult& result);
napi_value ToNapiValue(napi_env env, const ModelStatusResult& result);
napi_value ToNapiValue(napi_env env, const OfficialSmokeResult& result);
napi_value StringArrayToNapiValue(napi_env env, const std::vector<std::string>& values);

} // namespace internvl
