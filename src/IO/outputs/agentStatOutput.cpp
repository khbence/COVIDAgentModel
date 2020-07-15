#include "agentStatOutput.h"
#include <limits>
#include "dynamicPPState.h"

AgentStatOutput::AgentStatOutput(const thrust::host_vector<AgentStats>& data) {
    rapidjson::Value stats(rapidjson::kArrayType);
    const auto& names = DynamicPPState::getStateNames();
    unsigned idx = 0;
    for (const auto& e : data) {
        if (e.infectedTimestamp != std::numeric_limits<decltype(e.infectedTimestamp)>::max()) {
            rapidjson::Value currentAgent(rapidjson::kObjectType);
            currentAgent.AddMember("ID", idx, allocator);
            currentAgent.AddMember("infectionTime", e.infectedTimestamp, allocator);
            currentAgent.AddMember("InfectionLoc", e.infectedLocation, allocator);
            currentAgent.AddMember("diagnosisTime", e.diagnosedTimestamp, allocator);
            rapidjson::Value worst(rapidjson::kObjectType);
            worst.AddMember("name", stringToObject(names[e.worstState]), allocator);
            worst.AddMember("begin", e.worstStateTimestamp, allocator);
            if (e.worstStateEndTimestamp == 0) {
                worst.AddMember("end", -1, allocator);
            } else {
                worst.AddMember("end", e.worstStateEndTimestamp, allocator);
            }
            currentAgent.AddMember("worstState", worst, allocator);
            stats.PushBack(currentAgent, allocator);
        }
        ++idx;
    }
    d.AddMember("Statistics", stats, allocator);
}