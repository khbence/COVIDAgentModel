#include "timeHandler.h"

[[nodiscard]] std::vector<Days> parseDays(const std::string& rawDays) {
    std::string day;
    std::vector<Days> result;
    std::transform(rawDays.begin(), rawDays.end(), std::back_inserter(day), [](char c) {
        return std::toupper(c);
    });
    if (day == "ALL") {
        result = decltype(result){ Days::MONDAY,
            Days::TUESDAY,
            Days::WEDNESDAY,
            Days::THURSDAY,
            Days::FRIDAY,
            Days::SATURDAY,
            Days::SUNDAY };
    } else if (day == "WEEKDAYS") {
        result = decltype(result){
            Days::MONDAY,
            Days::TUESDAY,
            Days::WEDNESDAY,
            Days::THURSDAY,
            Days::FRIDAY,
        };
    } else if (day == "WEEKENDS") {
        result = decltype(result){ Days::SATURDAY, Days::SUNDAY };
    } else if (day == "MONDAY") {
        result.push_back(Days::MONDAY);
    } else if (day == "TUESDAY") {
        result.push_back(Days::TUESDAY);
    } else if (day == "WEDNESDAY") {
        result.push_back(Days::WEDNESDAY);
    } else if (day == "THURSDAY") {
        result.push_back(Days::THURSDAY);
    } else if (day == "FRIDAY") {
        result.push_back(Days::FRIDAY);
    } else if (day == "SATURDAY") {
        result.push_back(Days::SATURDAY);
    } else if (day == "SUNDAY") {
        result.push_back(Days::SUNDAY);
    } else {
        throw IOAgentTypes::InvalidDayInSchedule(rawDays);
    }
    return result;
}

Timehandler::Timehandler(unsigned timeStep_p, unsigned weeksInTheFuture)
    : timeStep(std::chrono::minutes(timeStep_p)),
      current(nextMidnight() + std::chrono::hours(hoursPerWeek * weeksInTheFuture)),
      stepsPerDay(minsPerDay / timeStep_p) {
    if (minsPerDay % timeStep_p != 0) { throw init::BadTimeStep(timeStep_p); }
}