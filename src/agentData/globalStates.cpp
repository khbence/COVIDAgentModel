#include "globalStates.h"

namespace states {
    SIRD& operator++(SIRD& e) { return e = static_cast<SIRD>(static_cast<int>(e) + 1); }

    [[nodiscard]] states::WBStates parseWBState(const std::string& rawState) {
        if (rawState.length() != 1) { throw IOAgentTypes::InvalidWBStateInSchedule(rawState); }
        char s = static_cast<char>(std::toupper(rawState.front()));
        switch (s) {
        case 'W':
            return states::WBStates::W;
        case 'N':
            return states::WBStates::N;
        case 'M':
            return states::WBStates::M;
        case 'S':
            return states::WBStates::S;
        case 'D':
            return states::WBStates::D;
        default:
            throw IOAgentTypes::InvalidWBStateInSchedule(rawState);
        }
    }
}// namespace states
