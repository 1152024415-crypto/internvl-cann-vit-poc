#include "rawfile_loader.h"

#include "rawfile/raw_file_manager.h"

#include <cstddef>
#include <cstring>

namespace internvl {

RawfileReadResult ReadRawfile(napi_env env, napi_value resourceManager, const std::string& fileName)
{
    RawfileReadResult result;

    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    if (nativeResourceManager == nullptr) {
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Failed to initialize NativeResourceManager";
        return result;
    }

    RawFile* rawFile = OH_ResourceManager_OpenRawFile(nativeResourceManager, fileName.c_str());
    if (rawFile == nullptr) {
        result.errorStage = "missing_artifact";
        result.errorMessage = "Missing rawfile: " + fileName;
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
        return result;
    }

    const long rawFileSize = OH_ResourceManager_GetRawFileSize(rawFile);
    if (rawFileSize <= 0) {
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Rawfile is empty or invalid: " + fileName;
        OH_ResourceManager_CloseRawFile(rawFile);
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
        return result;
    }

    result.bytes.resize(static_cast<std::size_t>(rawFileSize));
    const int readSize = OH_ResourceManager_ReadRawFile(rawFile, result.bytes.data(), rawFileSize);
    OH_ResourceManager_CloseRawFile(rawFile);
    OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);

    if (static_cast<long>(readSize) != rawFileSize) {
        result.bytes.clear();
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Rawfile read size mismatch: " + fileName;
        return result;
    }

    result.ok = true;
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
