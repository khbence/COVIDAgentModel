#pragma once
#include "globalStates.h"
#include <string>
#include "transitionMatrix.h"
#include "progressionMatrixFormat.h"

class DynamicPPState {
    char state = 0;// a number
    float infectious = 0.0;
    bool susceptible = true;

    short daysBeforeNextState = -1;

    static HD SingleBadTransitionMatrix& getTransition();

    void HD updateMeta();

public:
    static void initTransitionMatrix(parser::TransitionFormat& inputData);
    static HD unsigned getNumberOfStates();

    explicit DynamicPPState(const std::string& name);
    void HD gotInfected();
    void HD update(float scalingSymptons);
    [[nodiscard]] char HD getStateIdx() const { return state; }
    [[nodiscard]] float HD isInfectious() const { return infectious; }
    [[nodiscard]] bool HD isSusceptible() const { return susceptible; }
};