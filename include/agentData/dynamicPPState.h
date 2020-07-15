#pragma once
#include "globalStates.h"
#include <string>
#include "progressionMatrices.h"
#include "progressionMatrixFormat.h"
#include "agentsList.h"
#include <vector>
#include <map>

using ProgressionMatrix = MultiBadMatrix;

class DynamicPPState {
    float infectious = 0.0;

    char state = 0;// a number
    short daysBeforeNextState = -1;
    bool susceptible = true;

    static HD ProgressionMatrix& getTransition();

    void HD updateMeta();

public:
    DynamicPPState() = default;
    static std::string initTransitionMatrix(parser::TransitionFormat& inputData);
    static HD unsigned getNumberOfStates();
    static std::vector<std::string> getStateNames();

    explicit DynamicPPState(const std::string& name);
    explicit DynamicPPState() {};
    void HD gotInfected();
    void HD update(float scalingSymptons, AgentStats& agentStats, unsigned simTime);
    [[nodiscard]] char HD getStateIdx() const { return state; }
    [[nodiscard]] states::WBStates HD getWBState() const;
    [[nodiscard]] float HD isInfectious() const { return infectious; }
    [[nodiscard]] bool HD isSusceptible() const { return susceptible; }
};
