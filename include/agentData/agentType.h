#pragma once
#include <string>
#include "agentTypesFormat.h"
#include <vector>
#include "globalStates.h"
#include "timeHandler.h"
#include "timeDay.h"
#include "unordered_map"
#include "datatypes.h"

class AgentTypeList {
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
    // use the getOffsetIndex function to get the position for this vector, and use this value to search in events
    thrust::device_vector<unsigned> eventOffset;
    thrust::device_vector<Event> events;

    [[nodiscard]] static unsigned getOffsetIndex(unsigned ID, states::WBStates state, Days day);

public:
    AgentTypeList() = default;
    AgentTypeList(std::size_t n);
    void addSchedule(unsigned ID, std::pair<states::WBStates, Days> state, const std::vector<Event>& schedule);
};