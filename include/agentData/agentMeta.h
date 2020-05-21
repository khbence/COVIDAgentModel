#pragma once
#include <map>
#include <array>
#include <utility>
#include <vector>
#include <string>
#include "parametersFormat.h"

class BasicAgentMeta {
    struct AgeInterval {
        unsigned from;
        unsigned to;
        float symptons;
        float transmission;

        explicit AgeInterval(parser::Parameters::Age in);
        bool operator==(unsigned age) { return (from < age) && (age < to); }
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