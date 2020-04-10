#pragma once
#include "agentsList.h"

template<typename PPState, typename Location>
class Agent {
    unsigned id;
    static inline AgentList<PPState, Location>* agentList = AgentList<PPState, Location>::getInstance();

public:
};