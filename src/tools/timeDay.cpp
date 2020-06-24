#include "timeDay.h"
#include <cmath>
#include <limits>

TimeDay::TimeDay(float raw)
    : hours(static_cast<decltype(hours)>(raw)), minutes(static_cast<decltype(minutes)>(std::round(((raw - static_cast<int>(raw)) / 0.6) * 60))) {
    if (raw == -1.0) { hours = std::numeric_limits<decltype(hours)>::max(); }
}

TimeDayDuration::TimeDayDuration(float raw) : TimeDay(raw) {}

bool TimeDayDuration::isUndefinedDuration() const { return hours == std::numeric_limits<decltype(hours)>::max(); }