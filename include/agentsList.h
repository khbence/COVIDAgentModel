#pragma once
#include <vector>
/*
template<typename T>
concept PPStateType = requires (T x) { x.update(); x.gotInfected(); };
*/

// singleton
template<typename T>
class Agent;

template<typename PPState, typename Location>
class AgentList {
    std::vector<PPState> PPValues;
    std::vector<bool> diagnosed;
    std::vector<Location*> locations;

    AgentList() = default;
    friend class Agent<AgentList>;

public:
    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    [[nodiscard]] bool checkConsistency() const;// if all vector are of the same lengths

    void addAgent(PPState state, bool isDiagnosed, Location* location) {
        // Or should we just trust push_back? I would trust it, or probably best would be if we should write the numbers in the input file
        if (PPValues.size() == PPValues.capacity()) {
            PPValues.reserve(PPValues.size() * 1.5 + 1);
            diagnosed.reserve(PPValues.size() * 1.5 + 1);
            locations.reserve(PPValues.size() * 1.5 + 1);
        }
        PPValues.push_back(state);
        diagnosed.push_back(isDiagnosed);
        locations.push_back(location);
        // Add this agent to the location provided
        location->addAgent(PPValues.size() - 1);
    }

    PPState& getPPState(unsigned i) { return PPValues[i]; }
};
