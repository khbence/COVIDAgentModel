#include <iostream>
#include "simulation.h"
#include "movementPolicies.h"
#include "infectionPolicies.h"
#include "agentMeta.h"
#include "PPStateTypes.h"
#include "programParameters.h"
// for testing
#include <inputJSON.h>
#include <random>
#include "randomGenerator.h"
#include <omp.h>
#include "timing.h"

using PositionType = int;
using TypeOfLocation = int;

int main(int argc, char const* argv[]) {
    BEGIN_PROFILING("main");
    constexpr unsigned lengthInWeeks = 12;
    constexpr unsigned timeStep = 10;
    RandomGenerator::init(omp_get_max_threads());
    Simulation<PositionType,
        TypeOfLocation,
        PPStateSIRextended,
        BasicAgentMeta,
        DummyMovement,
        BasicInfection>
        s(PARSE_PARAMETERS(argc, argv, ProgramParameters));


    s.initialization();
    s.runSimulation();
    END_PROFILING("main");
    Timing::report();
    return EXIT_SUCCESS;
}
