#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cassert>

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
    explicit Timehandler(unsigned timeStep_p, unsigned weeksInTheFuture = 0)
        : timeStep(std::chrono::minutes(timeStep_p)),
          current(nextMidnight() + std::chrono::hours(hoursPerWeek * weeksInTheFuture)),
          stepsPerDay(minsPerDay / timeStep_p) {
        assert(minsPerDay % timeStep_p == 0);
    }

    bool operator<(const Timehandler& rhs) { return current < rhs.current; }

    Timehandler& operator++() {
        current += timeStep;
        ++counter;
        return *this;
    }

    [[nodiscard]] bool isMidnight() const { return (counter % stepsPerDay) == 0; }

    [[nodiscard]] unsigned getStepsPerDay() const {return stepsPerDay;}

    friend std::ostream& operator<<(std::ostream& out, const Timehandler& t) {
        auto t_c = std::chrono::system_clock::to_time_t(t.current);
        out << std::put_time(std::localtime(&t_c), "%F %T");
        return out;
    }

    void printDay() const {
        auto t_c = std::chrono::system_clock::to_time_t(current);
        std::cout << std::put_time(std::localtime(&t_c), "%F\n");
    }
};