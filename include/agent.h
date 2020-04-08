#pragma once
#include "agentsList.h"

template<typename PPState, typename Location>
class Agent {
    int id;
    static inline AgentList<PPState, Location>* agentList =
        AgentList<PPState, Location>::getInstance();

public:
    //I don't know yet, what access function are going to be need
};