#include "agentType.h"

AgentType::AgentType(std::string&& name_p) : name(std::move(name_p)) {}

void AgentType::addSchedule(std::pair<states::WBStates, Days> state,
    const std::vector<AgentType::Event>& schedule) {
    schedules.emplace(state, schedule);
}