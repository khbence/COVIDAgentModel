#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>

// TODO update after C++20
template<unsigned timeStepMin>
class Timehandler {
    std::chrono::system_clock::time_point current = std::chrono::system_clock::now();
    std::chrono::minutes timeStep = std::chrono::minutes(timeStepMin);

    static constexpr unsigned hoursPerWeek = 168;
    static constexpr unsigned minsPerDay = 1440;
    static_assert(minsPerDay % timeStepMin == 0);

    unsigned counter = 0;
    const unsigned stepsPerDay = minsPerDay / timeStepMin;

    auto nextMidnight() {
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
    template<unsigned>
    friend class Timehandler;

    Timehandler() : current(nextMidnight()) {}

    explicit Timehandler(unsigned weeksInTheFuture)
        : current(nextMidnight() + std::chrono::hours(hoursPerWeek * weeksInTheFuture)) {}

    template<unsigned M>
    bool operator<(const Timehandler<M>& rhs) {
        return current < rhs.current;
    }

    Timehandler& operator++() {
        current += timeStep;
        ++counter;
        return *this;
    }

    [[nodiscard]] bool isMidnight() const { return (counter % stepsPerDay) == 0; }

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