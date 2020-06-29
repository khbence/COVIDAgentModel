#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include "customExceptions.h"
#include "timeDay.h"
#include "datatypes.h"

class TimeDay;

enum class Days { MONDAY = 0, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY };

// TODO update after C++20
class Timehandler {
    std::chrono::system_clock::time_point current = std::chrono::system_clock::now();
    std::chrono::minutes timeStep;

    static constexpr unsigned hoursPerWeek = 168;
    static constexpr unsigned minsPerDay = 1440;

    unsigned counter = 0;
    const unsigned stepsPerDay;

    static auto nextMidnight() {
        auto now = std::chrono::system_clock::now();
        time_t tnow = std::chrono::system_clock::to_time_t(now);
        tm* date = std::localtime(&tnow);
        date->tm_hour = 0;
        date->tm_min = 0;
        date->tm_sec = 0;
        ++date->tm_mday;
        auto midnight = std::chrono::system_clock::from_time_t(std::mktime(date));
        return midnight;
    }

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

    Timehandler HD operator+(unsigned steps) const;
    Timehandler& HD operator+=(unsigned steps);
    Timehandler HD operator+(const TimeDayDuration& dur) const;
    Timehandler& HD operator+=(const TimeDayDuration& dur);

    Timehandler HD operator-(unsigned steps) const;
    Timehandler& HD operator-=(unsigned steps);
    Timehandler HD operator-(const TimeDayDuration& dur) const;
    Timehandler& HD operator-=(const TimeDayDuration& dur);

    [[nodiscard]] static std::vector<Days> parseDays(const std::string& rawDays);

    explicit Timehandler(unsigned timeStep_p, unsigned weeksInTheFuture = 0);

    bool HD operator<(const Timehandler& rhs) { return current < rhs.current; }
    bool HD operator>(const Timehandler& rhs) { return current > rhs.current; }

    Timehandler& operator++() {
        current += timeStep;
        ++counter;
        return *this;
    }

    unsigned HD getStepsUntilMidnight() const;
    Timehandler& HD getNextMidnight() const;

    [[nodiscard]] bool isMidnight() const { return (counter % stepsPerDay) == 0; }

    [[nodiscard]] unsigned getStepsPerDay() const { return stepsPerDay; }

    friend std::ostream& operator<<(std::ostream& out, const Timehandler& t) {
        auto t_c = std::chrono::system_clock::to_time_t(t.current);
        out << std::put_time(std::localtime(&t_c), "%F %T");
        return out;
    }

    Days HD getDay() const;// TODO calculate day "type"

    void printDay() const {
        auto t_c = std::chrono::system_clock::to_time_t(current);
        std::cout << std::put_time(std::localtime(&t_c), "%F\n");
    }
};