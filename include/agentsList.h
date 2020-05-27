#pragma once
#include <vector>
#include "datatypes.h"
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
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;

    using PPState_t = PPState;

    AgentList() = default;
    friend class Agent<AgentList>;

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    [[nodiscard]] bool checkConsistency() const;// if all vector are of the same lengths

    void initializeWithNumAgents(unsigned numAgents) {
        PPValues.resize(numAgents);
        diagnosed.resize(numAgents);
        location.resize(numAgents);
        agentMetaData.resize(numAgents);
    }
    void setAgent(unsigned index, PPState state, bool isDiagnosed, unsigned agentLocation) {
        PPValues[index] = state;
        diagnosed[index] = isDiagnosed;
        location[index] = agentLocation;
    }
    void setAgents(std::vector<PPState> &_states, std::vector<bool> &_diagnosed, std::vector<unsigned> &_location) {
        PPValues = _states;
        diagnosed = _diagnosed;
        location = _location;
        /*thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(agents.begin(),thrust::counting_iterator<int>(0))),
                         thrust::make_zip_iterator(thrust::make_tuple(agents.end(),thrust::counting_iterator<int>(agents.size()))),
                        ,[]HD(thrust::tuple<Agent<AgentList>&, int&> tup){thrust::get<0>(tup).id = thrust::get<1>(tup);});*/
    }
    unsigned addAgent(PPState state, bool isDiagnosed, unsigned agentLocation) {
        // Or should we just trust push_back? I would trust it, or probably best would be if we
        // should write the numbers in the input file
        if (PPValues.size() == PPValues.capacity()) {
            diagnosed.reserve(PPValues.size() * 1.5 + 10);
            location.reserve(PPValues.size() * 1.5 + 10);
            agentMetaData.reserve(PPValues.size() * 1.5 + 10);

            // This has to be the last one!
            PPValues.reserve(PPValues.size() * 1.5 + 10);
        }
        PPValues.push_back(state);
        diagnosed.push_back(isDiagnosed);
        location.push_back(agentLocation);
        agentMetaData.push_back(AgentMeta());
        // Add this agent to the location provided
        return PPValues.size() - 1;
    }

    PPState& getPPState(unsigned i) { return PPValues[i]; }
};
