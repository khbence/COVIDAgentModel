#include "basicLengthAbstract.h"

BasicLengthAbstract::LengthOfState::LengthOfState(int avgLength_p, int maxLength_p)
     : avgLength(avgLength_p), maxLength(maxLength_p), p(1.0 / static_cast<double>(avgLength_p)) {
    if (maxLength == -1) { maxLength = std::numeric_limits<decltype(maxLength)>::max(); }
}

// Note: [0, maxLength), because the 0 will run for a day, so the maxLength would run for
// maxLength+1 days
[[nodiscard]] HD int BasicLengthAbstract::LengthOfState::calculateDays() const {
    if (avgLength == -1) { return -1; }
    int days = RandomGenerator::geometric(p);
    while (maxLength < days) { days = RandomGenerator::geometric(p); }
    return days;
}