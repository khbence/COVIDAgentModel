#include "agentMeta.h"
#include "customExceptions.h"
#include "JSONDecoder.h"
#include <limits>

BasicAgentMeta::AgeInterval::AgeInterval(parser::Parameters::Age in)
    : symptons(in.symptoms), transmission(in.transmission) {
    if (in.from < 0) { throw IOParameters::NegativeFrom(); }
    from = in.from;
    if (in.to < 0) {
        to = std::numeric_limits<decltype(to)>::max();
    } else {
        to = in.to;
    }
}

std::array<std::pair<char, float>, 2> BasicAgentMeta::sexScaling;
std::vector<BasicAgentMeta::AgeInterval> BasicAgentMeta::ageScaling;
std::map<unsigned, float> BasicAgentMeta::preConditionScaling;

// init the three static variable with the data that coming from parameters json file
void BasicAgentMeta::initData(const std::string& inputFile) {
    auto input = DECODE_JSON_FILE(inputFile, parser::Parameters);
    // init the scaling based in sex
    if (input.sex.size() != 2) { throw IOParameters::NotBinary(); }
    for (unsigned i = 0; i < sexScaling.size(); ++i) {
        if (!(input.sex[0].name == "F" || input.sex[0].name == "M")) {
            throw IOParameters::WrongGenderName();
        }
        sexScaling[i] = std::make_pair(input.sex[i].name[0], input.sex[i].symptoms);
    }
    if (sexScaling[0].first == sexScaling[1].first) { throw IOParameters::WrongGenderName(); }

    // init the scalings based on age
    ageScaling.reserve(input.age.size());
    for (auto ageInter : input.age) { ageScaling.emplace_back(ageInter); }

    // init the scaling that coming from pre-conditions
    for (const auto& cond : input.preCondition) {
        preConditionScaling.emplace(std::make_pair(cond.ID, cond.symptoms));
    }
}

float BasicAgentMeta::getScalingSymptoms() const { return scalingSymptoms; }

float BasicAgentMeta::getScalingTransmission() const { return scalingTransmission; }