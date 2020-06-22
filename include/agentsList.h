#pragma once
#include <vector>
#include "datatypes.h"
#include <string>
#include "agentType.h"
#include <map>
#include "parametersFormat.h"
#include "agentTypesFormat.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "timeHandler.h"
#include <iterator>
#include "agentsFormat.h"
#include "agentMeta.h"
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
        agents.reserve(s);
    }

public:
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;
    thrust::device_vector<unsigned> types;
    thrust::device_vector<std::vector<unsigned>> possibleLocations;

    using PPState_t = PPState;

    friend class Agent<AgentList>;

    thrust::device_vector<Agent<AgentList>> agents;

    void initAgentMeta(const parser::Parameters& data) { AgentMeta::initData(data); }

    [[nodiscard]] std::map<unsigned, unsigned> initAgentTypes(const parser::AgentTypes& inputData) {
        // For the runtime performance, it would be better, that the IDs of the agent types would be
        // the same as their indexes, but we can not ensure it in the input file, so I create this
        // mapping, that will be used by the agents when I fill them up. Use it only during
        // initialization ID from files -> index in vectors
        std::map<unsigned, unsigned> agentTypeIDMapping;

        // agent types
        agentTypes.reserve(inputData.types.size());
        unsigned idx = 0;
        for (auto& type : inputData.types) {
            agentTypeIDMapping.emplace(type.ID, idx++);
            AgentType currentAgentType{ type.name };
            for (const auto& sch : type.schedulesUnique) {
                auto wb = states::parseWBState(sch.WB);
                auto days = Timehandler::parseDays(sch.dayType);

                std::vector<AgentType::Event> events;
                events.reserve(sch.schedule.size());
                for (const auto& e : sch.schedule) { events.emplace_back(e); }
                for (auto day : days) { currentAgentType.addSchedule(std::make_pair(wb, day), events); }
            }

            agentTypes.push_back(currentAgentType);
        }

        return agentTypeIDMapping;
    }

    void initAgents(const parser::Agents& inputData, const std::map<unsigned, unsigned>& locMap, const std::map<unsigned, unsigned>& typeMap) {
        reserve(inputData.people.size());
        for (const auto& person : inputData.people) {
            PPValues.push_back(PPState{ person.state });
            if (person.sex.size() != 1) { throw IOAgents::InvalidGender(person.sex); }
            agentMetaData.push_back(BasicAgentMeta(person.sex.front(), person.age, person.preCond));
            // I don't know if we should put any data about it in the input
            diagnosed.push_back(false);
            // Where to put them first?
            location.push_back(0);
            agents.push_back(Agent<AgentList>{ static_cast<unsigned>(agents.size()) });
            auto itType = typeMap.find(person.typeID);
            if (itType == typeMap.end()) { throw IOAgents::InvalidAgentType(person.typeID); }
            types.push_back(itType->second);
            std::vector<unsigned> locs;
            locs.reserve(person.locations.size());
            for (const auto& l : person.locations) {
                auto itLoc = locMap.find(l.locID);
                if (itLoc == locMap.end()) { throw IOAgents::InvalidLocationID(l.locID); }
                locs.push_back(itLoc->second);
            }
            possibleLocations.push_back(locs);
        }
    }

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    PPState& getPPState(unsigned i) { return PPValues[i]; }
};
