#pragma once
#include "agentsList.h"
#include "globalStates.h"

template<typename AgentListType>
class Agent {
    unsigned id;
    static inline AgentListType* agentList = AgentListType::getInstance();

public:
    explicit Agent(unsigned id_p) : id(id_p) {}
    [[nodiscard]] states::SIRD getSIRDState() const { return agentList->PPValues[id].getSIRD(); }
    [[nodiscard]] auto& getPPState() const { return agentList->PPValues[id]; }
    void gotInfected() { agentList->PPValues[id].gotInfected(); }
    void progressDisease(float additionalFactor = 1.0) {
        float scalingFactor = additionalFactor * agentList->agentMetaData[id].getScalingSymptoms();
        agentList->PPValues[id].update(scalingFactor);
    }
};