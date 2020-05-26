#pragma once
#include <vector>
#include "datatypes.h"
#include <string>
#include "agentType.h"
#include <map>
#include "agentTypesFormat.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "timeHandler.h"
#include <iterator>
#include "agentsFormat.h"
/*
template<typename T>
concept PPStateType = requires (T x) { x.update(); x.gotInfected(); };
*/

template<typename T>
class Agent;

template<typename PPState, typename AgentMeta, typename Location>
class AgentList {
    AgentList() = default;

    thrust::device_vector<AgentType> agentTypes;

    void reserve(std::size_t s) {
        PPValues.reserve(s);
        agentMetaData.reserve(s);
        diagnosed.reserve(s);
        location.reserve(s);
    }

public:
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;

    using PPState_t = PPState;

    friend class Agent<AgentList>;

    thrust::device_vector<Agent<AgentList>> agents;

    void initAgentMeta(const std::string& parametersFile) { AgentMeta::initData(parametersFile); }

    [[nodiscard]] std::map<unsigned, unsigned> initAgentTypes(const std::string& agentTypesFile) {
        auto input = DECODE_JSON_FILE(agentTypesFile, parser::AgentTypes);

        // For the runtime performance, it would be better, that the IDs of the agent types would be
        // the same as their indexes, but we can not ensure it in the input file, so I create this
        // mapping, that will be used by the agents when I fill them up. Use it only during
        // initialization ID from files -> index in vectors
        std::map<unsigned, unsigned> agentTypeIDMapping;

        // common schedules
        std::vector<std::vector<AgentType::Event>> commonSchedules;
        commonSchedules.reserve(input.commonSchedules.size());
        for (const auto& sch : input.commonSchedules) {
            if (sch.ID != commonSchedules.size()) {
                throw IOAgentTypes::BadIDCommonSchedules(sch.ID);
            }
            std::vector<AgentType::Event> current;
            current.reserve(sch.schedule.size());
            for (const auto& e : sch.schedule) { current.emplace_back(e); }
            commonSchedules.emplace_back(std::move(current));
        }

        // agent types
        agentTypes.reserve(input.types.size());
        unsigned idx = 0;
        for (auto& type : input.types) {
            agentTypeIDMapping.emplace(type.ID, idx++);
            AgentType currentAgentType{ std::move(type.name) };
            for (const auto& sch : type.schedulesUnique) {
                auto wb = states::parseWBState(sch.WB);
                auto days = parseDays(sch.dayType);

                std::vector<AgentType::Event> events;
                events.reserve(sch.schedule.size());
                for (const auto& e : sch.schedule) { events.emplace_back(e); }
                for (auto day : days) {
                    currentAgentType.addSchedule(std::make_pair(wb, day), events);
                }
            }

            for (const auto& sch : type.schedulesTypic) {
                auto wb = states::parseWBState(sch.WB);
                auto days = parseDays(sch.dayType);
                for (auto day : days) {
                    currentAgentType.addSchedule(
                        std::make_pair(wb, day), commonSchedules[sch.scheduleID]);
                }
            }

            agentTypes.push_back(currentAgentType);
        }

        return agentTypeIDMapping;
    }

    void initAgents(const std::string& agentsFile) {
        auto input = DECODE_JSON_FILE(agentsFile, parser::Agents);
    }

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    thrust::device_vector<Agent<AgentList>>& getAgentsList() { return agents; }

    PPState& getPPState(unsigned i) { return PPValues[i]; }
};
