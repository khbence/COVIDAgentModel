#include "PPStateTypes.h"

//static stuff
namespace detail {
    namespace PPStateSIRextended {
        unsigned numberOfStates = 1 + 6 + 3 + 1;// S + I + R + D
        std::array<unsigned, 5> startingIdx{ 0,
        1,
        7,
        10,
        11 };// to convert from idx to state
        SingleBadTransitionMatrix *transition;
    }
}

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
// PPStateSIRextended::SingleBadTransitionMatrix<PPStateSIRextended::numberOfStates> transition;

void PPStateSIRextended::applyNewIdx() {
    state = states::SIRD::S;
    for (int i = 0; i < 4; i++) {
        if (idx >= detail::PPStateSIRextended::startingIdx[i] && idx < detail::PPStateSIRextended::startingIdx[i + 1]) {
            state = (states::SIRD)i;
            subState = idx - detail::PPStateSIRextended::startingIdx[i];
        }
    }
}

SingleBadTransitionMatrix& PPStateSIRextended::getTransition() {return *detail::PPStateSIRextended::transition;};

void PPStateSIRextended::printHeader() {
    // I was lazy to do it properly
    std::cout << "S, I1, I2, I3, I4, I5, I6, R1, R2, R3, D\n";
}

PPStateSIRextended::PPStateSIRextended() : PPStateSIRAbstract(states::SIRD::S) {}
PPStateSIRextended::PPStateSIRextended(states::SIRD s) : PPStateSIRAbstract(s) {
    idx = static_cast<char>(state);
    daysBeforeNextState = getTransition().calculateJustDays(idx);
}
PPStateSIRextended::PPStateSIRextended(char idx_p)
    : PPStateSIRAbstract(states::SIRD::S), idx(idx_p) {
    applyNewIdx();
    daysBeforeNextState = getTransition().calculateJustDays(idx);
}

void PPStateSIRextended::gotInfected() {
    idx = 1;
    applyNewIdx();
    daysBeforeNextState = -2;
    // std::cout << "From " << 0 << " -> " << (int)idx<<"\n";
}

void PPStateSIRextended::update(float scalingSymptons) {
    // the order of the first two if is intentional
    if (daysBeforeNextState == -2) { daysBeforeNextState = getTransition().calculateJustDays(idx); }
    if (daysBeforeNextState > 0) { --daysBeforeNextState; }
    if (daysBeforeNextState == 0) {
        auto tmp = getTransition().calculateNextState(idx, scalingSymptons);
        auto stateIdx = tmp.first;
        auto days = tmp.second;
        daysBeforeNextState = days;
        idx = stateIdx;
        applyNewIdx();
    }
}

char PPStateSIRextended::getStateIdx() const { return idx; }

void PPStateSIRextended::initTransitionMatrix(const std::string& inputFile) {
        detail::PPStateSIRextended::transition = new SingleBadTransitionMatrix(inputFile);
}

unsigned PPStateSIRextended::getNumberOfStates() {
    return detail::PPStateSIRextended::numberOfStates;
}
