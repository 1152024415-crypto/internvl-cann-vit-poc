#include "rawfile_loader.h"

#include "rawfile/raw_file_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace internvl {

RawfileReadResult ReadRawfile(NativeResourceManager* nativeResourceManager, const std::string& fileName)
{
    RawfileReadResult result;

    if (nativeResourceManager == nullptr) {
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Failed to initialize NativeResourceManager";
        return result;
    }

    RawFile64* rawFile = OH_ResourceManager_OpenRawFile64(nativeResourceManager, fileName.c_str());
    if (rawFile == nullptr) {
        result.errorStage = "missing_artifact";
        result.errorMessage = "Missing rawfile: " + fileName;
        return result;
    }

    const int64_t rawFileSize = OH_ResourceManager_GetRawFileSize64(rawFile);
    if (rawFileSize <= 0) {
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Rawfile is empty or invalid: " + fileName;
        OH_ResourceManager_CloseRawFile64(rawFile);
        return result;
    }

    result.bytes.resize(static_cast<std::size_t>(rawFileSize));
    std::size_t offset = 0;
    while (offset < result.bytes.size()) {
        const std::size_t remaining = result.bytes.size() - offset;
        const std::size_t chunkSize = std::min<std::size_t>(remaining, 16 * 1024 * 1024);
        const int64_t readSize = OH_ResourceManager_ReadRawFile64(
            rawFile, result.bytes.data() + offset, static_cast<int64_t>(chunkSize));
        if (readSize <= 0) {
            result.bytes.clear();
            result.errorStage = "read_rawfile_failed";
            result.errorMessage = "Rawfile read failed before EOF: " + fileName;
            OH_ResourceManager_CloseRawFile64(rawFile);
            return result;
        }
        offset += static_cast<std::size_t>(readSize);
    }
    OH_ResourceManager_CloseRawFile64(rawFile);

    if (offset != static_cast<std::size_t>(rawFileSize)) {
        result.bytes.clear();
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Rawfile read size mismatch: " + fileName;
        return result;
    }

    result.ok = true;
    return result;
}

RawfileReadResult ReadRawfile(napi_env env, napi_value resourceManager, const std::string& fileName)
{
    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    RawfileReadResult result = ReadRawfile(nativeResourceManager, fileName);
    if (nativeResourceManager != nullptr) {
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
    }
    return result;
}

std::vector<float> BytesToFp32Vector(const std::vector<std::uint8_t>& bytes)
{
    std::vector<float> values(bytes.size() / sizeof(float));
    if (!values.empty()) {
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    }
    return values;
}

} // namespace internvl
