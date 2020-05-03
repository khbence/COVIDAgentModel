#pragma once
#include <vector>
#include "globalStates.h"
#include "agent.h"
#include <cmath>
#include <algorithm>
#include <random>
#include "randomGenerator.h"
#include "statistics.h"
#include "datatypes.h"

// concept
template<typename SimulationType>
class Location {
    using AgentType = Agent<typename SimulationType::AgentListType>;

    typename SimulationType::PositionType_t position;
    typename SimulationType::TypeOfLocation_t locType;
    thrust::device_vector<unsigned> agents;
    Statistic<typename SimulationType::PPState_t, AgentType> stat;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    thrust::device_vector<unsigned>& getAgents() { return agents; }

    void addAgent(unsigned a) {
        agents.push_back(a);
        stat.refreshStatisticNewAgent(a);
    }

    void removeAgent(unsigned idx) {
        // agents.back().swap(agents[idx]);
        thrust::swap(agents.back(), agents[idx]);
        stat.refreshStatisticRemoveAgent(agents.back());
        agents.pop_back();
    }

    // TODO optimise randoms for performance
    void infectAgents(double ratio) {
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        int newInfections = thrust::transform_reduce(
            thrust::make_permutation_iterator(ppstates.begin(), agents.begin()),
            thrust::make_permutation_iterator(ppstates.begin(), agents.end()),
            [&](auto& a) {
                if (a.getSIRD() == states::SIRD::S && RandomGenerator::randomUnit() < ratio) {
                    a.gotInfected();
                    return 1;
                }
                return 0;
            },
            0,
            thrust::plus<int>());
    }

    const auto& refreshAndGetStatistic() { return stat.refreshandGetAfterMidnight(agents); }
};