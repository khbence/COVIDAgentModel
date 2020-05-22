#pragma once

class BasicAgentMeta {
    float scalingSymptoms = 1.0;
    float scalingTransmission = 1.0;

    // TODO constructor which calculates this from the input data
public:
    [[nodiscard]] HD float getScalingSymptoms() const { return scalingSymptoms; }
    [[nodiscard]] HD float getScalingTransmission() const { return scalingTransmission; }
};