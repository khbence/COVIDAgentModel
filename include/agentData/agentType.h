#pragma once
#include <string>
#include "agentTypesFormat.h"
#include <vector>
#include "globalStates.h"
#include "timeHandler.h"
#include "unordered_map"

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
        float start;
        float end;

    public:
        // template because there are multiple (two) different kind of event, they all have these
        // used fields
        template<typename EventType>
        explicit Event(const EventType& in)
            : locationType(in.type), chance(static_cast<float>(in.chance)), start(static_cast<float>(in.start)), end(static_cast<float>(in.end)) {}
    };

private:
    std::string name;
    // TODO use some thrust version
    std::unordered_map<std::pair<states::WBStates, Days>, std::vector<Event>> schedules;

public:
    explicit AgentType(const std::string& name_p);
    void addSchedule(std::pair<states::WBStates, Days> state, const std::vector<Event>& schedule);
};