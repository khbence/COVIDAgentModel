#pragma once
#include "datatypes.h"

namespace states {
    enum class SIRD { S = 0, I, R, D };
    HD SIRD& operator++(SIRD& e);
    enum class WBStates { W = 0, N, M, S, D };
}// namespace states