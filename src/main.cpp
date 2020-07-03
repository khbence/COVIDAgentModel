#include <array>
#include "progressionMatrices.h"
#include "simulation.h"
#include "dynamicPPState.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include <iostream>
#include "agentMeta.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"
#include <omp.h>
#include "timing.h"
#include <cxxopts.hpp>
#include "smallTools.h"
#include "datatypes.h"

cxxopts::Options defineProgramParameters() {
    cxxopts::Options options("covid", "An agent-based epidemic simulator");
    options.add_options()("w,weeks", "Length of simulation in weeks", cxxopts::value<unsigned>()->default_value("12"))("t,deltat",
        "Length of timestep in minutes",
        cxxopts::value<unsigned>()->default_value("10"))("n,numagents", "Number of agents", cxxopts::value<int>()->default_value("-1"))(
        "N,numlocs", "Number of dummy locations", cxxopts::value<int>()->default_value("-1"))("P,progression",
        "Path to the progression matrix JSON file",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "progression.json"))("a,agents",
        "Agents file, for all human being in the experiment.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "agents.json"))("A,agentTypes",
        "List and schedule of all type fo agents.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "agentTypes.json"))("l,locations",
        "List of all locations in the simulation.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "locations.json"))("L,locationTypes",
        "List of all type of locations",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "locationTypes.json"))("p,parameters",
        "List of all general parameters for the simulation except the progression data.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "parameters.json"))("c,configRandom",
        "Config file for random initialization.",
        cxxopts::value<std::string>()->default_value(".." + separator() + "inputFiles" + separator() + "configRandom.json"));
    ;

    return options;
}

using PositionType = std::array<double, 2>;
using TypeOfLocation = unsigned;
using ProgressionMatrix = MultiBadMatrix;
using PPStates = DynamicPPState;
using Simulation_t = Simulation<PositionType, TypeOfLocation, PPStates, BasicAgentMeta, DummyMovement, BasicInfection>;

int main(int argc, char** argv) {
    BEGIN_PROFILING("main");

    auto options = defineProgramParameters();
    Simulation_t::addProgramParameters(options);

    options.add_options()("h,help", "Print usage");
    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help") != 0) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    RandomGenerator::init(omp_get_max_threads());
    Simulation_t s{ result };

    s.runSimulation();
    END_PROFILING("main");
    Timing::report();
    return EXIT_SUCCESS;
}
