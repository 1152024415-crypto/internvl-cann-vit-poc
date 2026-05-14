#pragma once

#include "napi/native_api.h"
#include "rawfile/raw_file_manager.h"

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

RawfileReadResult ReadRawfile(NativeResourceManager* nativeResourceManager, const std::string& fileName);
RawfileReadResult ReadRawfile(napi_env env, napi_value resourceManager, const std::string& fileName);
std::vector<float> BytesToFp32Vector(const std::vector<std::uint8_t>& bytes);

} // namespace internvl
