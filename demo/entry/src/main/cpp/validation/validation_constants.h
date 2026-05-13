#pragma once

#include <cstddef>
#include <cstdint>

namespace internvl {

constexpr const char* kModelFile = "internvl3_5_vit_projector_fp32_opset18_staticpos.om";

constexpr std::size_t kInputN = 1;
constexpr std::size_t kInputC = 3;
constexpr std::size_t kInputH = 448;
constexpr std::size_t kInputW = 448;
constexpr std::size_t kInputElementCount = kInputN * kInputC * kInputH * kInputW;
constexpr std::size_t kInputByteCount = kInputElementCount * sizeof(float);

constexpr std::size_t kOutputN = 1;
constexpr std::size_t kOutputTokenCount = 256;
constexpr std::size_t kOutputHiddenSize = 1024;
constexpr std::size_t kOutputElementCount = kOutputN * kOutputTokenCount * kOutputHiddenSize;
constexpr std::size_t kOutputByteCount = kOutputElementCount * sizeof(float);

constexpr double kCosineMin = 0.999;
constexpr double kMeanAbsDiffMax = 0.01;

struct TestCase {
    const char* name;
    const char* inputFile;
    const char* expectedOutputFile;
    const char* metadataFile;
};

constexpr TestCase kTestCases[] = {
    {"dog", "dog_pixel_values_fp32.bin", "dog_visual_tokens_fp32.bin", "dog.metadata.json"},
    {"cat", "cat_pixel_values_fp32.bin", "cat_visual_tokens_fp32.bin", "cat.metadata.json"},
};

} // namespace internvl
