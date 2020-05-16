#pragma once
#include "parser.h"
#include <string>
#include "smallTools.h"

struct InputParameters {
    BEGIN_PARAMETER_DECLARATION();

    ADD_PARAMETER_DEFAULT_VALUE_H(a,
        agents,
        std::string,
        ".." + separator() + "inputFiles" + separator() + "agents.json",
        "Agents file, for all human being in the experiment.");
    ADD_PARAMETER_DEFAULT_VALUE_H(A,
        agentsTypes,
        std::string,
        ".." + separator() + "inputFiles" + separator() + "agentsTypes.json",
        "List and schedule of all type fo agents.");

    ADD_PARAMETER_DEFAULT_VALUE_H(l,
        locations,
        std::string,
        ".." + separator() + "inputFiles" + separator() + "locations.json",
        "List of all locations in the simulation");

    ADD_PARAMETER_DEFAULT_VALUE_H(L,
        locationTypes,
        std::string,
        ".." + separator() + "inputFiles" + separator() + "locationTypes.json",
        "List of all type of locations");

    ADD_PARAMETER_DEFAULT_VALUE_H(p,
        parameters,
        std::string,
        ".." + separator() + "inputFiles" + separator() + "parameters.json",
        "List of all general parameters for the simulation except the progression data");

    ADD_PARAMETER_DEFAULT_VALUE_H(P,
        progression,
        std::string,
        ".." + separator() + "inputFiles" + separator() + "progression.json",
        "Progression matrix");

    ADD_PARAMETER_DEFAULT_VALUE_H(t,
        timeStep,
        unsigned,
        10,
        "Time step of the simulation in minutes");

    ADD_PARAMETER_DEFAULT_VALUE_H(w,
        weeks,
        unsigned,
        6,
        "Length of the simulation in weeks");

    END_PARAMETER_DECLARATION();
};