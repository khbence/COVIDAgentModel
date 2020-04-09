#pragma once
#include <vector>

template<typename PositionType, typename TypeOfLocation>
class Location {
    PositionType position;
    TypeOfLocation locType;
    std::vector<unsigned> agents;

public:
	Location(PositionType p, TypeOfLocation t) : position(p), locType(t) {}
	void addAgent(unsigned a) {
		agents.push_back(a);
	}
	std::vector<unsigned>& getAgents() {
		return agents;
	}
};