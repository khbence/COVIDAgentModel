#include "dataProvider.h"
#include "JSONDecoder.h"
#include <set>
#include <algorithm>

DataProvider::ProgressionType::ProgressionType(
    const parser::ProgressionDirectory::ProgressionFile& file)
    : ageBegin(file.age[0]), ageEnd(file.age[1]), preCond(file.preCond) {}

void DataProvider::readParameters(const std::string& fileName) {
    parameters = DECODE_JSON_FILE(fileName, decltype(parameters));
}

std::map<DataProvider::ProgressionType, std::string>
    DataProvider::readProgressionConfig(const std::string& fileName) {
    auto config = DECODE_JSON_FILE(fileName, parser::ProgressionDirectory);
    std::map<ProgressionType, std::string> progressions;
    for (const auto& f : config.transitionMatrices) {
        progressions.emplace(std::make_pair(f, f.fileName));
    }
    return progressions;
}

void DataProvider::readProgressionMatrices(const std::string& fileName) {
    auto rawProgressions = readProgressionConfig(fileName);
    std::vector<std::pair<unsigned, unsigned>> ageInters;
    for (const auto& kv : rawProgressions) {
        for (auto it = ageInters.begin(); it < ageInters.end(); ++it) {
            if (it->first == kv.first.ageEnd) {
                ageInters.insert(
                    it, std::make_pair(kv.first.ageBegin, kv.first.ageEnd));
                break;
            }
        }
    }
    auto numberCond = parameters.preCondition.size();
    for (const auto& kv : rawProgressions) {
        auto currentAgeBegin = kv.first.ageBegin;
        auto currentPreCond = kv.first.preCond;
        auto ageIndex = std::distance(ageInters.begin(),
            std::find_if(ageInters.begin(),
                ageInters.end(),
                [currentAgeBegin](
                    const auto& e) { return e.first == kv.first.ageBegin; }));

        auto condIndex = std::distance(parameters.preCondition.begin(),
            std::find_if(parameters.preCondition.begin(),
                parameters.preCondition.end(),
                [currentPreCond](
                    const auto& e) { return e.condition == currentPreCond; }));
        unsigned index = ageIndex * numberCond + numberCond;

        progressionDirectory.emplace(std::make_pair(kv.first,
            std::make_pair(
                DECODE_JSON_FILE(kv.second, parser::TransitionFormat), index)));
    }
}


void DataProvider::readConfigRandom(const std::string& fileName) {
    configRandom = DECODE_JSON_FILE(fileName, decltype(configRandom));
}

void DataProvider::readAgentTypes(const std::string& fileName) {
    agentTypes = DECODE_JSON_FILE(fileName, decltype(agentTypes));
    for (const auto& aType : agentTypes.types) {
        std::set<unsigned> locs{
            locationTypes.hospital,
            locationTypes.publicSpace,
            locationTypes.home,
            locationTypes.doctor
        };//, locationTypes.school, locationTypes.work };
        for (const auto& sch : aType.schedulesUnique) {
            for (const auto& event : sch.schedule) { locs.insert(event.type); }
        }
        aTypeToLocationTypes.emplace(std::piecewise_construct,
            std::forward_as_tuple(aType.ID),
            std::forward_as_tuple(locs.begin(), locs.end()));
    }
}

void DataProvider::readLocationTypes(const std::string& fileName) {
    locationTypes = DECODE_JSON_FILE(fileName, decltype(locationTypes));
}

void DataProvider::readLocations(const std::string& fileName,
    bool randomAgents) {
    locations = DECODE_JSON_FILE(fileName, decltype(locations));
    if (randomAgents) {
        for (const auto& l : locations.places) {
            typeToLocationMapping[l.type].push_back(l.ID);
        }
    }
}

void DataProvider::readAgents(const std::string& fileName) {
    agents = DECODE_JSON_FILE(fileName, decltype(agents));
}

void DataProvider::randomLocations(unsigned N) {
    locations.places.reserve(N);
    auto locTypes = static_cast<unsigned>(locationTypes.types.size());
    for (unsigned i = 0; i < N; ++i) {
        parser::Locations::Place current{};
        current.ID = std::to_string(i);
        current.type =
            randomSelect(configRandom.locationTypeDistibution.begin());
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
        current.typeID =
            randomSelect(configRandom.agentTypeDistribution.begin());
        const auto& requestedLocations = aTypeToLocationTypes[current.typeID];
        current.locations.reserve(requestedLocations.size());
        for (const auto& l : requestedLocations) {
            parser::Agents::Person::Location currentLoc{};
            currentLoc.typeID = l;
            const auto& possibleLocations =
                typeToLocationMapping[currentLoc.typeID];
            if ((possibleLocations.size() == 0)
                || (RandomGenerator::randomUnit()
                    < configRandom.irregulalLocationChance)) {
                currentLoc.locID = locations
                                       .places[RandomGenerator::randomUnsigned(
                                           locations.places.size())]
                                       .ID;
            } else {
                auto r =
                    RandomGenerator::randomUnsigned(possibleLocations.size());
                currentLoc.locID = possibleLocations[r];
            }
            current.locations.push_back(currentLoc);
        }
        agents.people.emplace_back(std::move(current));
    }
}

void DataProvider::randomStates() {
    for (auto& a : agents.people) {
        a.state = randomSelect(configRandom.stateDistibution.begin());
    }
}

DataProvider::DataProvider(const cxxopts::ParseResult& result) {
    PROFILE_FUNCTION();
    readParameters(result["parameters"].as<std::string>());
    readProgressionMatrices(result["progression"].as<std::string>());
    int numberOfAgents = result["numagents"].as<int>();
    int numberOfLocations = result["numlocs"].as<int>();
    if ((numberOfAgents != -1) || (numberOfLocations != -1)
        || result["randomStates"].as<bool>()) {
        readConfigRandom(result["configRandom"].as<std::string>());
    }
    readLocationTypes(result["locationTypes"].as<std::string>());
    readAgentTypes(result["agentTypes"].as<std::string>());
    if (numberOfLocations == -1) {
        readLocations(
            result["locations"].as<std::string>(), numberOfAgents == -1);
    } else {
        randomLocations(numberOfLocations);
    }
    if (numberOfAgents == -1) {
        readAgents(result["agents"].as<std::string>());
        if (result["randomStates"].as<bool>()) { randomStates(); }
    } else {
        randomAgents(numberOfAgents);
    }
}

[[nodiscard]] parser::Agents& DataProvider::acquireAgents() { return agents; }
[[nodiscard]] parser::AgentTypes& DataProvider::acquireAgentTypes() {
    return agentTypes;
}

[[nodiscard]] parser::Locations& DataProvider::acquireLocations() {
    return locations;
}

[[nodiscard]] parser::LocationTypes& DataProvider::acquireLocationTypes() {
    return locationTypes;
}

[[nodiscard]] parser::Parameters& DataProvider::acquireParameters() {
    return parameters;
}

[[nodiscard]] const std::map<DataProvider::ProgressionType,
    std::pair<parser::TransitionFormat, unsigned>>&
    DataProvider::acquireProgressionMatrices() {
    return progressionDirectory;
}

[[nodiscard]] const std::map<unsigned, std::vector<unsigned>>&
    DataProvider::getAgentTypeLocTypes() const {
    return aTypeToLocationTypes;
}
