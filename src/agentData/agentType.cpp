#include "agentType.h"
#include "customExceptions.h"

AgentTypeList::Event::Event() : locationType(0), chance(-1.0), start(0.0), end(0.0), duration(0.0) {}

AgentTypeList::Event::Event(const parser::AgentTypes::Type::ScheduleUnique::Event& in)
    : locationType(in.type),
      chance(static_cast<float>(in.chance)),
      start(static_cast<float>(in.start)),
      end(static_cast<float>(in.end)),
      duration(static_cast<float>(in.duration)) {}

unsigned AgentTypeList::getOffsetIndex(unsigned ID, states::WBStates state, Days day) {
    return (7 * static_cast<unsigned>(state)) + static_cast<unsigned>(day) + ID * 28;
}

AgentTypeList::AgentTypeList(std::size_t n) : eventOffset((n * 28) + 1, 0), events() {}

void AgentTypeList::addSchedule(unsigned ID, std::pair<states::WBStates, Days> state, const std::vector<AgentTypeList::Event>& schedule) {
    auto idx = getOffsetIndex(ID, state.first, state.second);
    auto n = schedule.size();
    for (auto i = idx + 1; i < eventOffset.size(); ++i) { eventOffset[i] += n; }
    events.insert(events.begin() + idx, schedule.begin(), schedule.end());
}