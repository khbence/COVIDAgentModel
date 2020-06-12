#include <iostream>
#include "simulation.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include "agentMeta.h"
#include "PPStateTypes.h"
#include "dynamicPPState.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"
#include <omp.h>
#include "timing.h"
#include <cxxopts.hpp>

using PositionType = std::array<double, 2>;
using TypeOfLocation = unsigned;
using PPState = DynamicPPState;

int main(int argc, char** argv) {
    BEGIN_PROFILING("main");
    cxxopts::Options options("covid", "An agent-based epidemic simulator");
    options.add_options()("l,length", "Length of simulation in weeks", cxxopts::value<unsigned>()->default_value("12"))("t,deltat",
        "Length of timestep in minutes",
        cxxopts::value<unsigned>()->default_value("10"))("n,numagents", "Number of agents", cxxopts::value<unsigned>()->default_value("100000"))(
        "i,infected", "Ratio of infected/exposed initially", cxxopts::value<double>()->default_value("0.01"))("I,infected2",
        "Ratio of infected 2 initially",
        cxxopts::value<double>()->default_value("0.0"))("numlocs", "Number of dummy locations", cxxopts::value<unsigned>()->default_value("1"))(
        "p,progression", "Path to the progression matrix JSON file", cxxopts::value<std::string>()->default_value("../inputFiles/transition.json"));

    RandomGenerator::init(omp_get_max_threads());
    Simulation<PositionType, TypeOfLocation, PPStateSIRextended, BasicAgentMeta, DummyMovement, BasicInfection> s{ PARSE_PARAMETERS(
        argc, argv, ProgramParameters) };


    s.initialization();
    s.runSimulation();
    END_PROFILING("main");
    Timing::report();
    return EXIT_SUCCESS;
}
