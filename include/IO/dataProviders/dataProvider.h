#pragma once
#include "agentsFormat.h"
#include "agentTypesFormat.h"
#include "locationsFormat.h"
#include "locationTypesFormat.h"
#include "parametersFormat.h"
#include "progressionConfigFormat.h"
#include "progressionMatrixFormat.h"
#include "configRandomFormat.h"
#include "cxxopts.hpp"

#include "randomGenerator.h"
#include <map>
#include <vector>
#include "timing.h"
#include <utility>

class DataProvider {
public:
    struct ProgressionType {
        unsigned ageBegin;
        unsigned ageEnd;
        std::string preCond;

        ProgressionType(
            const parser::ProgressionDirectory::ProgressionFile& file);
        
        friend bool operator<(const ProgressionType& lhs, const ProgressionType rhs) {
            if(lhs.ageBegin < rhs.ageBegin) {
                return lhs.preCond < rhs.preCond;
            }
            return false;
        }

        friend bool operator<(const ProgressionType& lhs, const std::pair<unsigned, std::string> rhs) {
            if(lhs.ageBegin < rhs.first) {
                return lhs.preCond < rhs.second;
            }
            return false;
        }

        friend bool operator<(const std::pair<unsigned, std::string> lhs, const ProgressionType& rhs) {
            if(lhs.first < rhs.ageBegin) {
                return lhs.second < rhs.preCond;
            }
            return false;
        }
    };

private:
    parser::Agents agents;
    parser::AgentTypes agentTypes;
    parser::Locations locations;
    parser::LocationTypes locationTypes;
    parser::Parameters parameters;
    std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>>
        progressionDirectory;

    // only for random generations and checking
    parser::ConfigRandom configRandom;
    std::map<unsigned, std::vector<unsigned>> aTypeToLocationTypes;
    std::map<unsigned, std::vector<std::string>> typeToLocationMapping;

    void readParameters(const std::string& fileName);
    std::map<ProgressionType, std::string> readProgressionConfig(
        const std::string& fileName);
    void readProgressionMatrices(const std::string& fileName);
    void readConfigRandom(const std::string& fileName);
    void readLocationTypes(const std::string& fileName);
    void readLocations(const std::string& fileName, bool randomAgents);
    void readAgentTypes(const std::string& fileName);
    void readAgents(const std::string& fileName);

    template<typename Iter>
    [[nodiscard]] auto randomSelect(Iter it) const {
        double r = RandomGenerator::randomUnit();
        double preSum = it->chance;
        while (preSum < r) {
            ++it;
            preSum += it->chance;
        }
        return it->value;
    }

    void randomLocations(unsigned N);
    void randomAgents(unsigned N);
    void randomStates();

public:
    explicit DataProvider(const cxxopts::ParseResult& result);

    [[nodiscard]] parser::Agents& acquireAgents();
    [[nodiscard]] parser::AgentTypes& acquireAgentTypes();
    [[nodiscard]] parser::Locations& acquireLocations();
    [[nodiscard]] parser::LocationTypes& acquireLocationTypes();
    [[nodiscard]] parser::Parameters& acquireParameters();
    [[nodiscard]] const std::map<ProgressionType,
        std::pair<parser::TransitionFormat, unsigned>>&
        acquireProgressionMatrices();

    [[nodiscard]] const std::map<unsigned, std::vector<unsigned>>&
        getAgentTypeLocTypes() const;
};
