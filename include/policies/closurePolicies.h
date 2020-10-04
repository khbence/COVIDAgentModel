#pragma once
#include <iostream>
#include "timeHandler.h"
#include "datatypes.h"
#include "cxxopts.hpp"
#include "operators.h"
#include "locationTypesFormat.h"

template<typename SimulationType>
class NoClosure {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data) {}

    void midnight(Timehandler simTime, unsigned timeStep) {}
    void step(Timehandler simTime, unsigned timeStep) {}
};

template<typename SimulationType>
class RuleClosure {
public:
    // add program parameters if we need any, this function got called already from Simulation
    static void addProgramParameters(cxxopts::Options& options) {}
    void initializeArgs(const cxxopts::ParseResult& result) {}
    void init(const parser::LocationTypes& data) {}

    void midnight(Timehandler simTime, unsigned timeStep) {}
    void step(Timehandler simTime, unsigned timeStep) {}
};