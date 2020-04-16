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
    std::vector<Agent<typename SimulationType::AgentListType>> agents;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}

    void addAgent(unsigned a) {
        const auto& newAgent = agents.emplace_back(a);
        Statistics::refreshStatisticNewAgent(newAgent);
    }

    std::vector<Agent<typename SimulationType::AgentListType>>& getAgents() {
        return agents;
    }

    // TODO this should be a policy, which we'll optimise for performance
    void infectAgents(double ratio) {
        unsigned newInfections = 0;
        // TODO random device and gen should be defined once
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        std::for_each(agents.begin(), agents.end(), [&](auto a) {
            if (a.getSIRDState() == states::SIRD::S && dis(gen) < ratio) {
                a.gotInfected();
                ++newInfections;
            }
        });
        Statistics::setNewlyInfected(newInfections);
    }
};