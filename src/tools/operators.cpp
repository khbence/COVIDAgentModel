#include "operators.h"
#include <tuple>

bool operator==(const Timehandler& lhs, const TimeDay& rhs) {
    unsigned minutes = lhs.counter * lhs.timeStep.count();
    return ((minutes / 60) == rhs.hours) && ((minutes % 60) == rhs.minutes);
}

bool operator==(const TimeDay& lhs, const Timehandler& rhs) { return rhs == lhs; }

bool operator!=(const Timehandler& lhs, const TimeDay& rhs) { return !(lhs == rhs); }

bool operator!=(const TimeDay& lhs, const Timehandler& rhs) { return !(lhs == rhs); }

bool operator<(const Timehandler& lhs, const TimeDay& rhs) {
    unsigned minutes = lhs.counter * lhs.timeStep.count();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(hours, mins) < std::tie(rhs.hours, rhs.minutes);
}

bool operator<(const TimeDay& lhs, const Timehandler& rhs) {
    unsigned minutes = rhs.counter * rhs.timeStep.count();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(lhs.hours, lhs.minutes) < std::tie(hours, mins);
}

bool operator>(const Timehandler& lhs, const TimeDay& rhs) {
    unsigned minutes = lhs.counter * lhs.timeStep.count();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(hours, mins) > std::tie(rhs.hours, rhs.minutes);
}

bool operator>(const TimeDay& lhs, const Timehandler& rhs) {
    unsigned minutes = rhs.counter * rhs.timeStep.count();
    const char hours = static_cast<char>(minutes / 60);
    const char mins = static_cast<char>(minutes % 60);
    return std::tie(lhs.hours, lhs.minutes) > std::tie(hours, mins);
}
