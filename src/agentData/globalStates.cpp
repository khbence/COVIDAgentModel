#include "globalStates.h"

namespace states {
    SIRD& operator++(SIRD& e) { return e = static_cast<SIRD>(static_cast<int>(e) + 1); }
}// namespace states
