#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>

template<unsigned timeStepMin>
class Timehandler {
    std::chrono::system_clock::time_point current = std::chrono::system_clock::now();
    std::chrono::minutes timeStep = std::chrono::minutes(timeStepMin);

    static constexpr unsigned hoursPerWeek = 168;

public:
    template<unsigned>
    friend class Timehandler;

    Timehandler() = default;

    explicit Timehandler(unsigned weeksInTheFuture)
        : current(std::chrono::system_clock::now() + std::chrono::hours(hoursPerWeek * weeksInTheFuture)) {}

    template<unsigned M>
    bool operator<(const Timehandler<M>& rhs) {
        return current < rhs.current;
    }

    Timehandler& operator++() {
        current += timeStep;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& out, const Timehandler& t) {
        auto t_c = std::chrono::system_clock::to_time_t(t.current);
        out << std::put_time(std::localtime(&t_c), "%F %T");
        return out;
    }
};