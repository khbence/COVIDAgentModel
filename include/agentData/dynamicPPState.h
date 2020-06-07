#pragma once
#include "globalStates.h"
#include <string>
#include "transitionMatrix.h"

class DynamicPPState {
    char state = 0;// a number
    bool infectious = false;
    bool susceptible = true;

    short daysBeforeNextState = -1;

    static HD SingleBadTransitionMatrix& getTransition();

    void HD updateMeta();

public:
    static void initTransitionMatrix(const std::string& inputFile);
    static HD unsigned getNumberOfStates();

    explicit DynamicPPState(const std::string& name);
    void HD gotInfected();
    void HD update(float scalingSymptons);
    [[nodiscard]] char HD getStateIdx() const { return state; }
    [[nodiscard]] bool HD isInfectious() const { return infectious; }
    [[nodiscard]] bool HD isSusceptible() const { return susceptible; }
};