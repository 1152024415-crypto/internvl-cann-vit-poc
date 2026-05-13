#pragma once

#include "napi/native_api.h"

#include <cstdint>
#include <string>
#include <vector>

namespace internvl {

struct RawfileReadResult {
    bool ok = false;
    std::vector<std::uint8_t> bytes;
    std::string errorStage;
    std::string errorMessage;
};

RawfileReadResult ReadRawfile(napi_env env, napi_value resourceManager, const std::string& fileName);
std::vector<float> BytesToFp32Vector(const std::vector<std::uint8_t>& bytes);

} // namespace internvl
