#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace internvl {

struct MetricsResult {
    bool ok = false;
    bool finite = false;
    std::size_t elementCount = 0;
    double maxAbsDiff = 0.0;
    double meanAbsDiff = 0.0;
    double cosine = 0.0;
    std::string errorStage;
    std::string errorMessage;
};

MetricsResult CompareFp32Tensors(const std::vector<float>& actual, const std::vector<float>& expected);

} // namespace internvl
