#pragma once
#include <vector>
/*
template<typename T>
concept PPStateType = requires (T x) { x.update(); x.gotInfected(); };
*/

// singleton
template<typename T>
class Agent;

template<typename PPState, typename AgentMeta, typename Location>
class AgentList {
    public:
    device_vector<PPState> PPValues;
    device_vector<AgentMeta> agentMetaData;
    device_vector<bool> diagnosed;
    device_vector<Location*> locations;

    using PPState_t = PPState;

    AgentList() = default;
    friend class Agent<AgentList>;

    device_vector<Agent<AgentList>> agents;


    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    [[nodiscard]] bool checkConsistency() const;// if all vector are of the same lengths

    void addAgent(PPState state, bool isDiagnosed, Location* location) {
        // Or should we just trust push_back? I would trust it, or probably best would be if we
        // should write the numbers in the input file
        /*if (PPValues.size() == PPValues.capacity()) {
            diagnosed.reserve(PPValues.size() * 1.5 + 10);
            locations.reserve(PPValues.size() * 1.5 + 10);
            agentMetaData.reserve(PPValues.size() * 1.5 + 10);
            agents.reserve(PPValues.size() * 1.5 + 10);

            // This has to be the last one!
            PPValues.reserve(PPValues.size() * 1.5 + 10);
        }*/
        PPValues.push_back(state);
        diagnosed.push_back(isDiagnosed);
        locations.push_back(location);
        agents.push_back(Agent<AgentList>(PPValues.size() - 1));
        agentMetaData.push_back(AgentMeta());
        // Add this agent to the location provided
        location->addAgent(PPValues.size() - 1);
    }

    device_vector<Agent<AgentList>>& getAgentsList() { return agents; }

    PPState& getPPState(unsigned i) { return PPValues[i]; }
};
