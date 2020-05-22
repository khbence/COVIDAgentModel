#include "globalStates.h"

namespace states {
    __host__ __device__ SIRD& operator++(SIRD& e) { return e = static_cast<SIRD>(static_cast<int>(e) + 1); }
}// namespace states
