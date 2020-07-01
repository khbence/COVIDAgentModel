#pragma once
#include "timeHandler.h"
#include "datatypes.h"

class Timehandler;
class TimeDayDuration;

class TimeDay {
protected:
    unsigned char hours;
    unsigned char minutes;

public:
    friend bool HD operator==(const Timehandler&, const TimeDay&);
    friend bool HD operator==(const TimeDay&, const Timehandler&);
    friend bool HD operator!=(const Timehandler&, const TimeDay&);
    friend bool HD operator!=(const TimeDay&, const Timehandler&);
    friend bool HD operator<(const Timehandler&, const TimeDay&);
    friend bool HD operator<=(const Timehandler&, const TimeDay&);
    friend bool HD operator<(const TimeDay&, const Timehandler&);
    friend bool HD operator<=(const TimeDay&, const Timehandler&);
    friend bool HD operator>(const Timehandler&, const TimeDay&);
    friend bool HD operator>=(const Timehandler&, const TimeDay&);
    friend bool HD operator>(const TimeDay&, const Timehandler&);
    friend bool HD operator>=(const TimeDay&, const Timehandler&);
    friend TimeDayDuration HD operator-(const TimeDay&, const Timehandler&);

    TimeDay HD operator+(const TimeDayDuration& dur) const;
    TimeDay& HD operator+=(const TimeDayDuration& dur);

    TimeDay HD operator-(const TimeDayDuration& dur) const;
    TimeDay& HD operator-=(const TimeDayDuration& dur);
    TimeDayDuration HD operator-(const TimeDay& other);



    explicit TimeDay(double raw);
    explicit TimeDay(unsigned mins);
    unsigned HD getMinutes() const;
};

class TimeDayDuration : public TimeDay {
public:
    explicit TimeDayDuration(double raw);
    explicit TimeDayDuration(unsigned mins);
    [[nodiscard]] bool HD isUndefinedDuration() const;
    [[nodiscard]] unsigned HD steps(unsigned timestep) const;
};