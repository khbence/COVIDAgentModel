#pragma once
#include <vector>

// concept
template<typename PositionType, typename TypeOfLocation, typename Statistics> class Location : public Statistics {
    PositionType position;
    TypeOfLocation locType;
    std::vector<unsigned> agents;

public:
    Location(PositionType p, TypeOfLocation t) : position(p), locType(t) {}
    void addAgent(unsigned a) { agents.push_back(a); }
    std::vector<unsigned>& getAgents() { return agents; }
};