#pragma once
#include <vector>
#include "globalStates.h"
#include "agent.h"
#include <cmath>
#include <algorithm>
#include <random>
#include "randomGenerator.h"
#include "statistics.h"

// concept
template<typename SimulationType>
class Location {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    typename SimulationType::PositionType_t position;
    typename SimulationType::TypeOfLocation_t locType;
    std::vector<AgentType> agents;
    Statistic<typename SimulationType::PPState_t, AgentType> stat;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    void addAgent(unsigned a) {
        const auto& newAgent = agents.emplace_back(a);
        stat.refreshStatisticNewAgent(newAgent);
    }

    void removeAgent(unsigned idx) {
        std::swap(agents[idx], agents.back());
        stat.refreshStatisticRemoveAgent(agents.back());
        agents.pop_back();
    }

    std::vector<AgentType>& getAgents() { return agents; }

    // TODO optimise randoms for performance
    void infectAgents(double ratio) {
        unsigned newInfections = 0;
        std::for_each(agents.begin(), agents.end(), [&](auto a) {
            if (a.getSIRDState() == states::SIRD::S && RandomGenerator::randomUnit() < ratio) {
                a.gotInfected();
                ++newInfections;
            }
        });
    }

    const auto& refreshAndGetStatistic() { return stat.refreshandGetAfterMidnight(agents); }
};