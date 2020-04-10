#pragma once
#include "agentsList.h"

template<typename AgentListType>
class Agent {
    unsigned id;
    static inline AgentListType* agentList = AgentListType::getInstance();

public:
    explicit Agent(unsigned id_p) : id(id_p) {}
};