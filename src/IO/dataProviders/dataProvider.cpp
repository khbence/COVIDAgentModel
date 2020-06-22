#include "dataProvider.h"
#include "JSONDecoder.h"
#include <set>

void DataProvider::readParameters(const std::string& fileName) { parameters = DECODE_JSON_FILE(fileName, decltype(parameters)); }

void DataProvider::readProgressionMatrix(const std::string& fileName) { progressionMatrix = DECODE_JSON_FILE(fileName, decltype(progressionMatrix)); }

void DataProvider::readConfigRandom(const std::string& fileName) { configRandom = DECODE_JSON_FILE(fileName, decltype(configRandom)); }

void DataProvider::readAgentTypes(const std::string& fileName) {
    agentTypes = DECODE_JSON_FILE(fileName, decltype(agentTypes));
    for (const auto& aType : agentTypes.types) {
        std::set<unsigned> locs{};
        for (const auto& sch : aType.schedulesUnique) {
            for (const auto& event : sch.schedule) { locs.insert(event.type); }
        }
        aTypeToLocationTypes.emplace(std::piecewise_construct, std::forward_as_tuple(aType.ID), std::forward_as_tuple(locs.begin(), locs.end()));
    }
}

void DataProvider::readLocationTypes(const std::string& fileName) { locationTypes = DECODE_JSON_FILE(fileName, decltype(locationTypes)); }

void DataProvider::readLocations(const std::string& fileName) { locations = DECODE_JSON_FILE(fileName, decltype(locations)); }

void DataProvider::readAgents(const std::string& fileName) { agents = DECODE_JSON_FILE(fileName, decltype(agents)); }

void DataProvider::randomLocations(unsigned N) {
    locations.places.reserve(N);
    auto locTypes = static_cast<unsigned>(locationTypes.types.size());
    for (unsigned i = 0; i < N; ++i) {
        parser::Locations::Place current{};
        current.ID = static_cast<decltype(current.ID)>(i);
        current.type = randomSelect(configRandom.locationTypeDistibution.begin());
        typeToLocationMapping[current.type].push_back(current.ID);
        current.coordinates = std::vector<double>{ 0.0, 0.0 };
        current.area = 0;
        current.state = "ON";
        current.capacity = 0;
        current.ageInter = std::vector<int>{ 0, 100 };
        locations.places.emplace_back(std::move(current));
    }
}

void DataProvider::randomAgents(unsigned N) {
    agents.people.reserve(N);
    for (unsigned i = 0; i < N; ++i) {
        parser::Agents::Person current{};
        current.age = RandomGenerator::randomUnsigned(90);
        current.sex = (RandomGenerator::randomUnit() < 0.5) ? "M" : "F";
        current.preCond = randomSelect(configRandom.preCondDistibution.begin());
        current.state = randomSelect(configRandom.stateDistibution.begin());
        current.typeID = randomSelect(configRandom.agentTypeDistribution.begin());
        const auto& requestedLocations = aTypeToLocationTypes[current.typeID];
        current.locations.reserve(requestedLocations.size());
        for (const auto& l : requestedLocations) {
            parser::Agents::Person::Location currentLoc{};
            currentLoc.typeID = l;
            const auto& possibleLocations = typeToLocationMapping[currentLoc.typeID];
            if ((possibleLocations.size() == 0) || (RandomGenerator::randomUnit() < configRandom.irregulalLocationChance)) {
                currentLoc.locID = locations.places[RandomGenerator::randomUnsigned(locations.places.size())].ID;
                assert(currentLoc.locID < locations.places.size());
            } else {
                auto r = RandomGenerator::randomUnsigned(possibleLocations.size());
                currentLoc.locID = possibleLocations[r];
                assert(currentLoc.locID < locations.places.size());
            }
            current.locations.push_back(currentLoc);
        }
        agents.people.emplace_back(std::move(current));
    }
}

DataProvider::DataProvider(const cxxopts::ParseResult& result) {
    readParameters(result["parameters"].as<std::string>());
    readProgressionMatrix(result["progression"].as<std::string>());
    int numberOfAgents = result["numagents"].as<int>();
    int numberOfLocations = result["numlocs"].as<int>();
    if ((numberOfAgents != -1) || (numberOfLocations != -1)) { readConfigRandom(result["configRandom"].as<std::string>()); }
    readAgentTypes(result["agentTypes"].as<std::string>());
    readLocationTypes(result["locationTypes"].as<std::string>());
    if (numberOfLocations == -1) {
        readLocations(result["locations"].as<std::string>());
    } else {
        randomLocations(numberOfLocations);
    }
    if (numberOfAgents == -1) {
        readAgents(result["agents"].as<std::string>());
    } else {
        randomAgents(numberOfAgents);
    }
}

[[nodiscard]] parser::Agents& DataProvider::acquireAgents() { return agents; }
[[nodiscard]] parser::AgentTypes& DataProvider::acquireAgentTypes() { return agentTypes; }
[[nodiscard]] parser::Locations& DataProvider::acquireLocations() { return locations; }
[[nodiscard]] parser::LocationTypes& DataProvider::acquireLocationTypes() { return locationTypes; }
[[nodiscard]] parser::Parameters& DataProvider::acquireParameters() { return parameters; }
[[nodiscard]] parser::TransitionFormat& DataProvider::acquireProgressionMatrix() { return progressionMatrix; }