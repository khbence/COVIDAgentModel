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
    device_vector<unsigned> agents;
    Statistic<typename SimulationType::PPState_t, AgentType> stat;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    device_vector<unsigned>& getAgents() {
        return agents;
    }

    void addAgent(unsigned a) {
        agents.push_back(a);
        stat.refreshStatisticNewAgent(a);
    }

    void removeAgent(unsigned idx) {
        //agents.back().swap(agents[idx]);
        swap(agents.back(), agents[idx]);
        stat.refreshStatisticRemoveAgent(agents.back());
        agents.pop_back();
    }

    // TODO optimise randoms for performance
    void infectAgents(double ratio) {
        // TODO random device and gen should be defined once
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        int newInfections = transform_reduce(make_permutation_iterator(ppstates.begin(), agents.begin()),
                 make_permutation_iterator(ppstates.begin(), agents.end()),[&](auto &a) {
            if (a.getSIRD() == states::SIRD::S && dis(gen) < ratio) {
                a.gotInfected();
                return 1;
            }
            return 0;
        },0,plus<int>());
    }

    const auto& refreshAndGetStatistic() { return stat.refreshandGetAfterMidnight(agents); }
};