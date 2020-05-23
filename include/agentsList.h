#pragma once
#include <vector>
#include "datatypes.h"
#include <string>
#include "agentType.h"
#include <map>
#include "agentTypesFormat.h"
/*
template<typename T>
concept PPStateType = requires (T x) { x.update(); x.gotInfected(); };
*/

template<typename T>
class Agent;

template<typename PPState, typename AgentMeta, typename Location>
class AgentList {
    AgentList() = default;

    thrust::device_vector<AgentType> agentTypes;

    // For the runtime performance, it would be better, that the IDs of the agent types would be the
    // same as their indexes, but we can not ensure it in the input file, so I create this mapping,
    // that will be used by the agents when I fill them up. Use it only during initialization ID
    // from files -> index in vectors
    std::map<unsigned, unsigned> agentTypeIDMapping;

public:
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;

    using PPState_t = PPState;

    friend class Agent<AgentList>;

    thrust::device_vector<Agent<AgentList>> agents;

    void initAgentMeta(const std::string& parametersFile) { AgentMeta::initData(parametersFile); }

    void initAgentTypes(const std::string& agentTypesFile) {
        auto input = DECODE_JSON_FILE(agentTypesFile, parser::AgentTypes);
    }

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    [[nodiscard]] bool checkConsistency() const;// if all vector are of the same lengths

    unsigned addAgent(PPState state, bool isDiagnosed, unsigned agentLocation) {
        // Or should we just trust push_back? I would trust it, or probably best would be if we
        // should write the numbers in the input file
        if (PPValues.size() == PPValues.capacity()) {
            diagnosed.reserve(PPValues.size() * 1.5 + 10);
            location.reserve(PPValues.size() * 1.5 + 10);
            agentMetaData.reserve(PPValues.size() * 1.5 + 10);
            agents.reserve(PPValues.size() * 1.5 + 10);

            // This has to be the last one!
            PPValues.reserve(PPValues.size() * 1.5 + 10);
        }
        PPValues.push_back(state);
        diagnosed.push_back(isDiagnosed);
        location.push_back(agentLocation);
        agents.push_back(Agent<AgentList>(PPValues.size() - 1));
        agentMetaData.push_back(AgentMeta());
        // Add this agent to the location provided
        return PPValues.size() - 1;
    }

    thrust::device_vector<Agent<AgentList>>& getAgentsList() { return agents; }

    PPState& getPPState(unsigned i) { return PPValues[i]; }
};
