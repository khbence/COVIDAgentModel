#include "PPStateSIRBasic.h"

void PPStateSIRBasic::update(/*elapsed time step + agent meta*/) {}

void PPStateSIRBasic::gotInfected() { this->state = states::SIRD::I; }

[[nodiscard]] states::SIRD PPStateSIRBasic::getSIRD() const { return this->state; }

[[nodiscard]] states::WBStates PPStateSIRBasic::getWBState() const {
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