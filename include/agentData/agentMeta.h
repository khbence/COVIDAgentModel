#pragma once

class BasicAgentMeta {
    float scalingSymptoms = 1.0;
    float scalingTransmission = 1.0;

    // TODO constructor which calculates this from the input data
public:
    [[nodiscard]] float getScalingSymptoms() const { return scalingSymptoms; }
    [[nodiscard]] float getScalingTransmission() const { return scalingTransmission; }
};