#include "PPStateTypes.h"

// Abstract
PPStateSIRAbstract::PPStateSIRAbstract(states::SIRD s) : state(s) {}

void PPStateSIRAbstract::gotInfected() { this->state = states::SIRD::I; }

[[nodiscard]] states::SIRD PPStateSIRAbstract::getSIRD() const { return this->state; }

[[nodiscard]] states::WBStates PPStateSIRAbstract::getWBState() const {
    switch (this->state) {
    case states::SIRD::R:
    case states::SIRD::S:
        return states::WBStates::W;
        break;
    case states::SIRD::I:
        return states::WBStates::N;
        break;
    case states::SIRD::D:
        return states::WBStates::D;
        break;
    default:
        return states::WBStates::W;
    }
}

// Basic
// TODO

// Extended
void PPStateSIRextended::applyNewIdx() {
    state = states::SIRD::S;
    for (const auto s : startingIdx) {
        if (s >= idx) {
            subState = static_cast<char>(idx - s);
            break;
        }
        ++state;
    }
}

PPStateSIRextended::PPStateSIRextended() : PPStateSIRAbstract(states::SIRD::S) {}
PPStateSIRextended::PPStateSIRextended(states::SIRD s) : PPStateSIRAbstract(s) {}
PPStateSIRextended::PPStateSIRextended(char idx_p)
    : PPStateSIRAbstract(states::SIRD::S), idx(idx_p) {
    applyNewIdx();
}

void PPStateSIRextended::update(float scalingSymptons) {
    if (daysBeforeNextState > 0) { --daysBeforeNextState; }
    if (daysBeforeNextState == 0) {
        auto [stateIdx, days] = transition.calculateNextState(idx, scalingSymptons);
        daysBeforeNextState = days;
        idx = stateIdx;
        applyNewIdx();
    }
}