#include "tensor_metrics.h"

#include "validation_constants.h"

#include <cmath>
#include <limits>

namespace internvl {

MetricsResult CompareFp32Tensors(const std::vector<float>& actual, const std::vector<float>& expected)
{
    MetricsResult result;
    result.elementCount = actual.size();

    if (actual.size() != expected.size()) {
        result.errorStage = "output_size_mismatch";
        result.errorMessage = "Actual and expected output element counts differ";
        return result;
    }

    if (actual.empty()) {
        result.errorStage = "compare_failed";
        result.errorMessage = "Output tensor is empty";
        return result;
    }

    double absDiffSum = 0.0;
    double maxAbsDiff = 0.0;
    double dot = 0.0;
    double actualNorm = 0.0;
    double expectedNorm = 0.0;
    bool finite = true;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double a = static_cast<double>(actual[index]);
        const double e = static_cast<double>(expected[index]);
        if (!std::isfinite(a) || !std::isfinite(e)) {
            finite = false;
        }
        const double diff = std::abs(a - e);
        absDiffSum += diff;
        if (diff > maxAbsDiff) {
            maxAbsDiff = diff;
        }
        dot += a * e;
        actualNorm += a * a;
        expectedNorm += e * e;
    }

    result.finite = finite;
    result.maxAbsDiff = maxAbsDiff;
    result.meanAbsDiff = absDiffSum / static_cast<double>(actual.size());
    if (actualNorm > 0.0 && expectedNorm > 0.0) {
        result.cosine = dot / (std::sqrt(actualNorm) * std::sqrt(expectedNorm));
    }

    if (!finite) {
        result.errorStage = "compare_failed";
        result.errorMessage = "Actual or expected output contains non-finite values";
        return result;
    }

    if (result.cosine < kCosineMin || result.meanAbsDiff > kMeanAbsDiffMax) {
        result.errorStage = "compare_failed";
        result.errorMessage = "Output metrics are outside validation thresholds";
        return result;
    }

    result.ok = true;
    return result;
}

} // namespace internvl
