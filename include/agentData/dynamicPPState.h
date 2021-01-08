#pragma once
#include "globalStates.h"
#include <string>
#include "progressionMatrices.h"
#include "progressionMatrixFormat.h"
#include "agentsList.h"
#include <vector>
#include <map>
#include "progressionType.h"

using ProgressionMatrix = MultiBadMatrix;

class DynamicPPState {
    float infectious = 0.0;
    unsigned progressionID = 0;

    char state = 0;// a number
    short daysBeforeNextState = -1;
    float susceptible = 1.0;

    static HD ProgressionMatrix& getTransition(unsigned progressionID_p);

    void HD updateMeta();

public:
    DynamicPPState() = default;
    static std::string initTransitionMatrix(
        std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
            inputData,
        parser::ProgressionDirectory& config);
    static HD unsigned getNumberOfStates();
    static std::vector<std::string> getStateNames();

    DynamicPPState(const std::string& name, unsigned progressionID_p);
    void HD gotInfected();
    bool HD update(float scalingSymptons,
        AgentStats& agentStats,
        unsigned simTime,
        unsigned agentID,
        unsigned tracked);
    [[nodiscard]] char HD getStateIdx() const { return state; }
    [[nodiscard]] states::WBStates HD getWBState() const;
    [[nodiscard]] float HD isInfectious() const { return infectious; }
    [[nodiscard]] float HD getSusceptible() const { return susceptible; }
    void HD setSusceptible(float s) { this->susceptible = s; }
    [[nodiscard]] bool HD isInfected() const;
    [[nodiscard]] char HD die(bool covid);
    [[nodiscard]] float HD getAccuracyPCR() const;
    [[nodiscard]] float HD getAccuracyAntigen() const;
};
