#pragma once
#include <string>
#include "agentTypesFormat.h"
#include <vector>
#include "globalStates.h"
#include "timeHandler.h"
#include "timeDay.h"
#include "unordered_map"
#include "datatypes.h"

// hash function for the key
namespace std {
    template<>
    class hash<std::pair<states::WBStates, Days>> {
    public:
        std::size_t operator()(std::pair<states::WBStates, Days> p) const {
            return std::hash<unsigned>()(static_cast<unsigned>(p.first)) ^ std::hash<unsigned>()(static_cast<unsigned>(p.second));
        }
    };
}// namespace std


class AgentType {
public:
    class Event {
        unsigned locationType;
        float chance;
        TimeDay start;
        TimeDay end;
        TimeDayDuration duration;

    public:
        Event();
        explicit Event(const parser::AgentTypes::Type::ScheduleUnique::Event& in);
    };

private:
    std::string name;
    // use the getOffsetIndex function to get the position for this vector, and use this value to search in events
    thrust::device_vector<unsigned> eventOffset;
    thrust::device_vector<Event> events;

    [[nodiscard]] static unsigned getOffsetIndex(states::WBStates state, Days day);

public:
    explicit AgentType(std::string name_p);
    void addSchedule(std::pair<states::WBStates, Days> state, const std::vector<Event>& schedule);
};