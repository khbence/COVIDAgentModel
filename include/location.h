#pragma once
#include <vector>
#include "globalStates.h"
#include "agent.h"
#include <cmath>
#include <algorithm>
#include <random>

// concept
template<typename SimulationType, typename Statistics>
class Location : public Statistics {
    typename SimulationType::PositionType_t position;
    typename SimulationType::TypeOfLocation_t locType;
    device_vector<unsigned> agents;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    void addAgent(unsigned a) {
        agents.push_back(a);
        //Statistics::refreshStatisticNewAgent<typename SimulationType::AgentListType>(a);
    }

    device_vector<unsigned>& getAgents() {
        return agents;
    }

    // TODO this should be a policy, which we'll optimise for performance
    void infectAgents(double ratio) {
        unsigned newInfections = 0;
        // TODO random device and gen should be defined once
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        auto& ppstates = SimulationType::AgentListType::getInstance()->PPValues;
        for_each(make_permutation_iterator(ppstates.begin(), agents.begin()),
                 make_permutation_iterator(ppstates.begin(), agents.end()),[&](auto &a) {
            if (a.getSIRD() == states::SIRD::S && dis(gen) < ratio) {
                a.gotInfected();
                ++newInfections;
            }
        });
        Statistics::setNewlyInfected(newInfections);
    }
};