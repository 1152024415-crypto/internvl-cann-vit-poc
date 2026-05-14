#include "napi_result.h"

#include <cstddef>

namespace internvl {

namespace {

void SetBool(napi_env env, napi_value object, const char* name, bool value)
{
    napi_value napiValue;
    napi_get_boolean(env, value, &napiValue);
    napi_set_named_property(env, object, name, napiValue);
}

void SetNumber(napi_env env, napi_value object, const char* name, double value)
{
    napi_value napiValue;
    napi_create_double(env, value, &napiValue);
    napi_set_named_property(env, object, name, napiValue);
}

void SetString(napi_env env, napi_value object, const char* name, const std::string& value)
{
    napi_value napiValue;
    napi_create_string_utf8(env, value.c_str(), value.size(), &napiValue);
    napi_set_named_property(env, object, name, napiValue);
}

} // namespace

napi_value ToNapiValue(napi_env env, const RunResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetString(env, object, "caseName", result.caseName);
    SetString(env, object, "deviceType", result.deviceType);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    SetNumber(env, object, "latencyMs", result.latencyMs);
    SetNumber(env, object, "outputElementCount", static_cast<double>(result.outputElementCount));
    SetNumber(env, object, "maxAbsDiff", result.maxAbsDiff);
    SetNumber(env, object, "meanAbsDiff", result.meanAbsDiff);
    SetNumber(env, object, "cosine", result.cosine);
    SetBool(env, object, "finite", result.finite);
    SetBool(env, object, "outputShapeOk", result.outputShapeOk);
    SetString(env, object, "outputShape", result.outputShape);
    return object;
}

napi_value ToNapiValue(napi_env env, const StabilityResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetString(env, object, "caseName", result.caseName);
    SetNumber(env, object, "repeatCount", static_cast<double>(result.repeatCount));
    SetNumber(env, object, "successCount", static_cast<double>(result.successCount));
    SetNumber(env, object, "minLatencyMs", result.minLatencyMs);
    SetNumber(env, object, "maxLatencyMs", result.maxLatencyMs);
    SetNumber(env, object, "avgLatencyMs", result.avgLatencyMs);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    napi_value lastRun = ToNapiValue(env, result.lastRun);
    napi_set_named_property(env, object, "lastRun", lastRun);
    return object;
}

napi_value ToNapiValue(napi_env env, const ModelStatusResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetBool(env, object, "loaded", result.loaded);
    SetString(env, object, "deviceType", result.deviceType);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    SetNumber(env, object, "latencyMs", result.latencyMs);
    return object;
}

napi_value ToNapiValue(napi_env env, const OfficialSmokeResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetString(env, object, "deviceType", result.deviceType);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    SetNumber(env, object, "latencyMs", result.latencyMs);
    SetNumber(env, object, "modelByteCount", static_cast<double>(result.modelByteCount));
    SetNumber(env, object, "inputCount", static_cast<double>(result.inputCount));
    SetNumber(env, object, "outputCount", static_cast<double>(result.outputCount));
    SetNumber(env, object, "inputByteCount", static_cast<double>(result.inputByteCount));
    SetNumber(env, object, "outputByteCount", static_cast<double>(result.outputByteCount));
    return object;
}

napi_value ToNapiValue(napi_env env, const OfficialClassificationResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetString(env, object, "imageName", result.imageName);
    SetString(env, object, "deviceType", result.deviceType);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    SetNumber(env, object, "latencyMs", result.latencyMs);
    SetNumber(env, object, "inputByteCount", static_cast<double>(result.inputByteCount));
    SetNumber(env, object, "outputElementCount", static_cast<double>(result.outputElementCount));
    SetString(env, object, "top1Label", result.top1Label);
    SetNumber(env, object, "top1Score", result.top1Score);
    SetString(env, object, "top2Label", result.top2Label);
    SetNumber(env, object, "top2Score", result.top2Score);
    SetString(env, object, "top3Label", result.top3Label);
    SetNumber(env, object, "top3Score", result.top3Score);
    return object;
}

napi_value StringArrayToNapiValue(napi_env env, const std::vector<std::string>& values)
{
    napi_value array;
    napi_create_array_with_length(env, values.size(), &array);
    for (std::size_t index = 0; index < values.size(); ++index) {
        napi_value item;
        napi_create_string_utf8(env, values[index].c_str(), values[index].size(), &item);
        napi_set_element(env, array, index, item);
    }
    return array;
}

} // namespace internvl
