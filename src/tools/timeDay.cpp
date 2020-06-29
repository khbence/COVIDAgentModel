#include "timeDay.h"
#include <cmath>
#include <limits>

TimeDay HD TimeDay::operator+(const TimeDayDuration& dur) const {
    TimeDay ret = *this;
    ret += dur;
    return ret;
}

TimeDay& HD TimeDay::operator+=(const TimeDayDuration& dur) {
    unsigned mins = dur.getMinutes();
    minutes += mins % 60;
    auto overflow = minutes / 60;
    minutes %= 60;
    hours += (mins / 60) + overflow;
    return *this;
}

TimeDay HD TimeDay::operator-(const TimeDayDuration& dur) const {
    TimeDay ret = *this;
    ret -= dur;
    return ret;
}

TimeDay& HD TimeDay::operator-=(const TimeDayDuration& dur) {
    auto minsDur = dur.getMinutes();
    auto minsThis = getMinutes();
    auto mins = minsThis - minsDur;
    minutes = mins % 60;
    hours = mins / 60;
    return *this;
}

TimeDay::TimeDay(float raw)
    : hours(static_cast<decltype(hours)>(raw)), minutes(static_cast<decltype(minutes)>(std::round(((raw - static_cast<int>(raw)) / 0.6) * 60))) {
    if (raw == -1.0) { hours = std::numeric_limits<decltype(hours)>::max(); }
}

unsigned HD TimeDay::getMinutes() const {
    return static_cast<unsigned>((static_cast<unsigned>(hours) * 60) + static_cast<unsigned>(minutes));
}

TimeDayDuration::TimeDayDuration(float raw) : TimeDay(raw) {}

bool HD TimeDayDuration::isUndefinedDuration() const { return hours == std::numeric_limits<decltype(hours)>::max(); }

[[nodiscard]] unsigned HD TimeDayDuration::steps(unsigned timestep) const {
    return ((unsigned)this->hours*60)/timestep +(unsigned)this->minutes/timestep;
};