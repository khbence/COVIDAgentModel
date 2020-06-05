#pragma once
#include "globalStates.h"
#include <string>
#include "transitionMatrix.h"

class DynamicPPState {
    char state;// a number

    // we can save the infectious and WB state here to not look it up in the global scope

    short daysBeforeNextState;

public:
    static void initTransitionMatrix(const std::string& inputFile);
    static HD unsigned getNumberOfStates();

    explicit HD DynamicPPState(const std::string& state);
    void HD gotInfected();
    void HD update(float scalingSymptons);
    [[nodiscard]] char HD getStateIdx() const;
};