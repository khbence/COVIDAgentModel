#pragma once

namespace states {
    enum class SIRD { S = 0, I, R, D };
    SIRD& operator++(SIRD& e);
    enum class WBStates { W = 0, N, M, S, D };
}// namespace states