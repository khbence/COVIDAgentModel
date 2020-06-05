#include "globalStates.h"
#include "customExceptions.h"

namespace states {
    __host__ __device__ SIRD& operator++(SIRD& e) {
        return e = static_cast<SIRD>(static_cast<int>(e) + 1);
    }

    [[nodiscard]] states::WBStates parseWBState(const std::string& rawState) {
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
            return states::WBStates::W;
        }
    }
}// namespace states
