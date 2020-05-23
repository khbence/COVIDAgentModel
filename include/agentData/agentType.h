#pragma once
#include <string>
#include "agentTypesFormat.h"

class AgentType {
    class Schedule {
        unsigned locationType;
        float chance;
        float from;
        float to;

    public:
        Schedule(const parser::AgentTypes::Schedule in);
    };

    std::string name;
};