#pragma once
#include "timeHandler.h"

class Timehandler;

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

    explicit TimeDay(float raw);
};

class TimeDayDuration : public TimeDay {
public:
    explicit TimeDayDuration(float raw);
    [[nodiscard]] bool isUndefinedDuration() const;
};