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
#include "agentStats.h"
#include "agentStatOutput.h"
#include "progressionMatrixFormat.h"
#include "dataProvider.h"
#include "progressionType.h"

template<typename T>
class Agent;

template<typename PPState, typename AgentMeta, typename Location>
class AgentList {
    AgentList() = default;

    void reserve(std::size_t s) {
        PPValues.reserve(s);
        agentMetaData.reserve(s);
        diagnosed.reserve(s);
        quarantined.reserve(s);
        location.reserve(s);
        agents.reserve(s);
    }

public:
    AgentTypeList agentTypes;
    thrust::device_vector<PPState> PPValues;
    thrust::device_vector<AgentMeta> agentMetaData;
    // id in the array of the progression matrices
    thrust::device_vector<bool> diagnosed;
    thrust::device_vector<unsigned> location;
    thrust::device_vector<unsigned> types;
    thrust::device_vector<AgentStats> agentStats;
    thrust::device_vector<bool> quarantined;

    thrust::device_vector<unsigned long> locationOffset;
    // longer, every agents' every locations, indexed by the offset
    thrust::device_vector<unsigned> possibleLocations;
    thrust::device_vector<unsigned> possibleTypes;

    using PPState_t = PPState;

    friend class Agent<AgentList>;

    thrust::device_vector<Agent<AgentList>> agents;

    void initAgentMeta(const parser::Parameters& data) { AgentMeta::initData(data); }

    [[nodiscard]] std::map<unsigned, unsigned> initAgentTypes(const parser::AgentTypes& inputData) {
        // For the runtime performance, it would be better, that the IDs of the
        // agent types would be the same as their indexes, but we can not ensure
        // it in the input file, so I create this mapping, that will be used by
        // the agents when I fill them up. Use it only during initialization ID
        // from files -> index in vectors
        std::map<unsigned, unsigned> agentTypeIDMapping;
        agentTypes = AgentTypeList(inputData.types.size());
        // agent types
        unsigned idx = 0;
        for (auto& type : inputData.types) {
            agentTypeIDMapping.emplace(type.ID, idx);
            for (const auto& sch : type.schedulesUnique) {
                auto wb = states::parseWBState(sch.WB);
                auto days = Timehandler::parseDays(sch.dayType);

                std::vector<AgentTypeList::Event> events;
                events.reserve(sch.schedule.size());
                for (const auto& e : sch.schedule) { events.emplace_back(e); }
                for (auto day : days) {
                    agentTypes.addSchedule(idx, std::make_pair(wb, day), events);
                }
            }
            ++idx;
        }

        return agentTypeIDMapping;
    }

    void initAgents(parser::Agents& inputData,
        const std::map<std::string, unsigned>& locMap,
        const std::map<unsigned, unsigned>& typeMap,
        const std::map<unsigned, std::vector<unsigned>>& agentTypeLocType,
        const std::map<ProgressionType, std::pair<parser::TransitionFormat, unsigned>, std::less<>>&
            progressionMatrices) {
        auto n = inputData.people.size();
        reserve(n);

        thrust::host_vector<PPState> PPValues_h;
        thrust::host_vector<AgentStats> agentStats_h;
        thrust::host_vector<AgentMeta> agentMetaData_h;
        thrust::host_vector<bool> diagnosed_h;
        thrust::host_vector<bool> quarantined_h;
        thrust::host_vector<unsigned> location_h;
        thrust::host_vector<unsigned> types_h;
        thrust::host_vector<Agent<AgentList>> agents_h;

        thrust::host_vector<unsigned long> locationOffset_h;
        // longer, every agents' every locations, indexed by the offset
        thrust::host_vector<unsigned> possibleLocations_h;
        thrust::host_vector<unsigned> possibleTypes_h;

        PPValues_h.reserve(n);
        agentMetaData_h.reserve(n);
        diagnosed_h.reserve(n);
        quarantined_h.reserve(n);
        location_h.reserve(n);
        types_h.reserve(n);
        agents_h.reserve(n);
        locationOffset_h.reserve(n + 1);
        locationOffset_h.push_back(0);
        agentStats_h.reserve(n);

        for (auto& person : inputData.people) {
            auto tmp = std::make_pair(static_cast<unsigned>(person.age), static_cast<std::string>(person.preCond));
            auto it = progressionMatrices.find(tmp);
            PPValues_h.push_back(PPState(person.state, it->second.second));
            AgentStats stat;
            // TODO: how do I tell that agent is infected (even if not
            // infectious)
            if (PPValues_h.back().getStateIdx() > 0) {// Is infected at the beginning
                stat.infectedTimestamp = 0;
                stat.worstState = PPValues_h.back().getStateIdx();
                stat.worstStateTimestamp = 0;
            }
            agentStats_h.push_back(stat);

            if (person.sex.size() != 1) { throw IOAgents::InvalidGender(person.sex); }
            agentMetaData_h.push_back(
                BasicAgentMeta(person.sex.front(), person.age, person.preCond));

            // I don't know if we should put any data about it in the input
            diagnosed_h.push_back(false);
            quarantined_h.push_back(false);
            // Where to put them first?
            location_h.push_back(0);
            agents_h.push_back(Agent<AgentList>{ static_cast<unsigned>(agents.size()) });

            // agentType
            auto itType = typeMap.find(person.typeID);
            if (itType == typeMap.end()) { throw IOAgents::InvalidAgentType(person.typeID); }
            types_h.push_back(itType->second);

            // locations
            const auto& requestedLocs = agentTypeLocType.find(person.typeID)->second;
            std::vector<bool> hasThatLocType(requestedLocs.size(), false);
            std::vector<unsigned> locs;
            std::vector<unsigned> ts;// types
            locs.reserve(person.locations.size());
            ts.reserve(person.locations.size());
            std::sort(person.locations.begin(),
                person.locations.end(),
                [](const auto& lhs, const auto& rhs) { return lhs.typeID < rhs.typeID; });
            for (const auto& l : person.locations) {
                auto itLoc = locMap.find(l.locID);
                if (itLoc == locMap.end()) { throw IOAgents::InvalidLocationID(l.locID); }
                locs.push_back(itLoc->second);
                ts.push_back(l.typeID);

                auto it = std::find(requestedLocs.begin(), requestedLocs.end(), l.typeID);
                if (it == requestedLocs.end()) {
                    throw IOAgents::UnnecessaryLocType(
                        agents_h.size() - 1, person.typeID, l.typeID);
                }
                hasThatLocType[std::distance(requestedLocs.begin(), it)] = true;
            }
            if (std::any_of(
                    hasThatLocType.begin(), hasThatLocType.end(), [](bool v) { return !v; })) {
                std::string missingTypes;
                for (unsigned idx = 0; idx < hasThatLocType.size(); ++idx) {
                    if (!hasThatLocType[idx]) {
                        missingTypes += std::to_string(requestedLocs[idx]) + ", ";
                    }
                }
                missingTypes.pop_back();
                missingTypes.pop_back();
                throw IOAgents::MissingLocationType(agents_h.size() - 1, std::move(missingTypes));
            }

            possibleLocations_h.insert(possibleLocations_h.end(), locs.begin(), locs.end());
            possibleTypes_h.insert(possibleTypes_h.end(), ts.begin(), ts.end());
            locationOffset_h.push_back(locationOffset_h.back() + locs.size());
        }

        PPValues = PPValues_h;
        agentMetaData = agentMetaData_h;
        diagnosed = diagnosed_h;
        quarantined = quarantined_h;
        location = location_h;
        types = types_h;
        agents = agents_h;
        locationOffset = locationOffset_h;
        possibleLocations = possibleLocations_h;
        possibleTypes = possibleTypes_h;
        agentStats = agentStats_h;
    }

    [[nodiscard]] static AgentList* getInstance() {
        static AgentList instance;
        return &instance;
    }

    PPState& getPPState(unsigned i) { return PPValues[i]; }

    void printAgentStatJSON(const std::string& fileName) {
        AgentStatOutput writer{ agentStats };
        writer.writeFile(fileName);
    }
};
