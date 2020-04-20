#pragma once

class BasicAgentMeta {
    float scalingSymptons = 1.0;
    float scalingTransmission = 1.0;

    // TODO constructor which calculates this from the input data
public:
    [[nodiscard]] float getScalingSymptons() const { return scalingSymptons; }
    [[nodiscard]] float getScalingTransmission() const { return scalingTransmission; }
};