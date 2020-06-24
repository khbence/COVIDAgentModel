#pragma once

class TimeDay {
protected:
    unsigned char hours;
    unsigned char minutes;

public:
    explicit TimeDay(float raw);
};

class TimeDayDuration : public TimeDay {
public:
    explicit TimeDayDuration(float raw);
    [[nodiscard]] bool isUndefinedDuration() const;
};