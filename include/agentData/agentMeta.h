#pragma once
#include <map>
#include <array>
#include <utility>
#include <vector>
#include <string>
#include "parametersFormat.h"

class BasicAgentMeta {
    class AgeInterval {
        unsigned from;
        unsigned to;
        float symptoms;
        float transmission;

    public:
        explicit AgeInterval(parser::Parameters::Age in);
        bool operator==(unsigned age) const { return (from < age) && (age < to); }
        [[nodiscard]] float getSymptoms() const;
        [[nodiscard]] float getTransmission() const;
    };

    float scalingSymptoms = 1.0;
    float scalingTransmission = 1.0;

    // As a good christian I hardcoded that there are only two genders
    static std::array<std::pair<char, float>, 2> sexScaling;
    static std::vector<AgeInterval> ageScaling;
    static std::map<unsigned, float> preConditionScaling;

public:
    static void initData(const std::string& inputFile);

    BasicAgentMeta(char gender, unsigned age, unsigned preCondition);

    [[nodiscard]] float getScalingSymptoms() const;
    [[nodiscard]] float getScalingTransmission() const;
};