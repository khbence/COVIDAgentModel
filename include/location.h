#pragma once
#include <vector>
#include "globalStates.h"
#include "agent.h"

// concept
template<typename SimulationType, typename Statistics>
class Location : public Statistics {
    typename SimulationType::PositionType_t position;
    typename SimulationType::TypeOfLocation_t locType;
    std::vector<Agent<typename SimulationType::AgentListType>> agents;

public:
    Location(decltype(position) p, decltype(locType) t) : position(p), locType(t) {}
    void addAgent(unsigned a) { agents.emplace_back(a); }
};