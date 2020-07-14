#include "agentStatOutput.h"
#include <limits>

AgentStatOutput::AgentStatOutput(const thrust::host_vector<AgentStats>& data) {
    rapidjson::Value stats(rapidjson::kArrayType);
    unsigned idx = 0;
    for (const auto& e : data) {
        if (e.infectedTimestamp != std::numeric_limits<decltype(e.infectedTimestamp)>::max()) {
            rapidjson::Value currentAgent(rapidjson::kObjectType);
            currentAgent.AddMember("ID", idx, allocator);
            currentAgent.AddMember("infectionTime", e.infectedTimestamp, allocator);
            currentAgent.AddMember("InfectionLoc", e.infectedLocation, allocator);
            currentAgent.AddMember("diagnosisTime", e.diagnosedTimestamp, allocator);
            rapidjson::Value worst(rapidjson::kObjectType);
            worst.AddMember("name", e.worstState, allocator);
            worst.AddMember("begin", e.worstStateTimestamp, allocator);
            worst.AddMember("end", e.worstStateEndTimestamp, allocator);
            currentAgent.AddMember("worstState", worst, allocator);
            stats.PushBack(currentAgent, allocator);
        }
        ++idx;
    }
    d.AddMember("Statistics", stats, allocator);
}