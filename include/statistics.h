#pragma once
#include "agent.h"
#include "globalStates.h"
#include <array>
#include <algorithm>

template<typename PPStateType, typename AgentType>
class Statistic {
    std::array<unsigned, PPStateType::numberOfStates> states;
    // we can store here nice combined stats, if we don't wanna calculate them all the time

public:
    void refreshStatisticNewAgent(const AgentType& a) {
        const auto& state = a.getPPState();
        ++states[state.getStateIdx()];
    }

    void refreshStatisticRemoveAgent(const AgentType& a) {
        const auto& state = a.getPPState();
        --states[state.getStateIdx()];
    }

    const decltype(states)& refreshandGetAfterMidnight(const std::vector<AgentType>& agents) {
        std::fill(states.begin(), states.end(), 0);
        std::for_each(agents.begin(), agents.end(), [&](const auto& a) {
            ++states[a.getPPState().getStateIdx()];
        });
        return states;
    }
};