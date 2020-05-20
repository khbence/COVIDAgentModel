#include <iostream>
#include "simulation.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include "agentMeta.h"
#include "PPStateTypes.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"
#include <omp.h>
#include "timing.h"

#include <cxxopts.hpp>


using PositionType = int;
using TypeOfLocation = int;

int main(int argc, char ** argv) {
    BEGIN_PROFILING("main");
    cxxopts::Options options("COVID Agent Model", "An agent-based epidemic simulator");
    options.add_options()
        ("l,length", "Length of simulation in weeks", cxxopts::value<unsigned>()->default_value("12"))
        ("t,deltat", "Length of timestep in minutes", cxxopts::value<unsigned>()->default_value("10"))
        ("n,numagents", "Number of agents", cxxopts::value<unsigned>()->default_value("100000"))
        ("i,infected", "Ratio of infected initially", cxxopts::value<double>()->default_value("0.01"))
        ("numlocs", "Number of dummy locations", cxxopts::value<unsigned>()->default_value("1"));
    
    RandomGenerator::init(omp_get_max_threads());
    Simulation<PositionType,
        TypeOfLocation,
        PPStateSIRextended,
        BasicAgentMeta,
        DummyMovement,
        BasicInfection>
        s (options);

    options.add_options()
        ("h,help", "Print usage");
    cxxopts::ParseResult result = options.parse(argc, argv);
    if (result.count("help"))
    {
      std::cout << options.help() << std::endl;
      exit(0);
    }
    s.initialize_args(result);

    unsigned lengthInWeeks = result["length"].as<unsigned>();
    unsigned timeStep = result["deltat"].as<unsigned>();

    // setup for test
    {
        unsigned numAgents = result["numagents"].as<unsigned>();
        double initial_infected_ratio = result["infected"].as<double>();
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        // Create basic locations for everyone
        unsigned numLocations = result["numlocs"].as<unsigned>();
        unsigned agentsPerLoc = numAgents/numLocations;
        
        for (int i = 0; i < numLocations; i++) s.addLocation(i, 0);

        // Populate agent list
        for (int i = 0; i < numAgents; i++) {
            s.addAgent(PPStateSIRextended(
                           dis(gen) < initial_infected_ratio ? states::SIRD::I : states::SIRD::S),
                false,
                i / agentsPerLoc);
        }
    }

    s.initialization();
    s.runSimulation(timeStep, lengthInWeeks);
    END_PROFILING("main");
    Timing::report();
    return EXIT_SUCCESS;
}
