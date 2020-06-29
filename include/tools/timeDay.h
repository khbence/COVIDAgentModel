#pragma once
#include "timeHandler.h"

class Timehandler;
class TimeDayDuration;

class TimeDay {
protected:
    unsigned char hours;
    unsigned char minutes;

public:
    friend bool operator==(const Timehandler&, const TimeDay&);
    friend bool operator==(const TimeDay&, const Timehandler&);
    friend bool operator!=(const Timehandler&, const TimeDay&);
    friend bool operator!=(const TimeDay&, const Timehandler&);
    friend bool operator<(const Timehandler&, const TimeDay&);
    friend bool operator<(const TimeDay&, const Timehandler&);
    friend bool operator>(const Timehandler&, const TimeDay&);
    friend bool operator>(const TimeDay&, const Timehandler&);

    TimeDay operator+(const TimeDayDuration& dur) const;
    TimeDay& operator+=(const TimeDayDuration& dur);

    explicit TimeDay(float raw);
    unsigned getMinutes() const;
};

class TimeDayDuration : public TimeDay {
public:
    explicit TimeDayDuration(float raw);
    [[nodiscard]] bool isUndefinedDuration() const;
};