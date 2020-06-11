#pragma once
#include "location.h"
#include "util.h"
#include <iostream>

template<class SimulationType>
class BasicInfection {
    struct Parameters {
        double v, h, s;
        double a, b;

        Parameters() = default;
    };

    Parameters par;

public:
    BasicInfection(cxxopts::Options& options) {
        options.add_options()("m,Imax", "Infection: [0 1] :max value ", cxxopts::value<double>()->default_value("1.0"))("v,Iasymmetry",
            "Infection: 1< :longer small phase [0 1] :longer high phase",
            cxxopts::value<double>()->default_value("1.0"))("H,Ihorizontal",
            "sigmoid: horizotal move of inflexcion point; <-1 or >1 :exponential like",
            cxxopts::value<double>()->default_value("0.0"))("s,Ishape", "shape: bigger, more steep", cxxopts::value<double>()->default_value("5.0"));
    }

protected:
    void initialize_args(cxxopts::ParseResult& result) {
        auto m = result["Imax"].as<double>() / 2;
        par.v = result["Iasymmetry"].as<double>();
        par.h = result["Ihorizontal"].as<double>();
        par.s = result["Ishape"].as<double>();
        double min = 1.0 / (1.0 + par.v * std::exp(-par.s * 2 * (0.0 - par.h - 0.5)));
        double max = 1.0 / (1.0 + par.v * std::exp(-par.s * 2 * (1.0 - par.h - 0.5)));

        par.a = m / (max - min);
        par.b = -m * min / (max - min);
    }

public:
    void infectionsAtLocations(unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationListOffsets =
            realThis->locs->locationListOffsets;// offsets into locationAgentList and locationIdsOfAgents
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        auto& ppstates = realThis->agents->PPValues;

        thrust::device_vector<double> infectionRatios(locationListOffsets.size() - 1, 0.0);
        thrust::device_vector<unsigned> fullInfectedCounts(locationListOffsets.size() - 1, 0);
        // Cout up infectious people - those who are Infected, and Infected state is >1
        reduce_by_location(locationListOffsets, fullInfectedCounts, ppstates, [] HD(typename SimulationType::PPState_t & ppstate) {
            return (unsigned)(ppstate.isInfectious());
        });
        auto parTMP = par;
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.begin(), locationListOffsets.begin(), locationListOffsets.begin() + 1)),
            thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.end(), locationListOffsets.end() - 1, locationListOffsets.end())),
            infectionRatios.begin(),
            [=] HD(thrust::tuple<unsigned&, unsigned&, unsigned&> tuple) {
                unsigned numInfectedAgentsPresent = thrust::get<0>(tuple);
                unsigned offset0 = thrust::get<1>(tuple);
                unsigned offset1 = thrust::get<2>(tuple);
                unsigned num_agents = offset1 - offset0;
                if (numInfectedAgentsPresent == 0) { return 0.0; }
                double densityOfInfected = static_cast<double>(numInfectedAgentsPresent) / num_agents;
                double y = 1.0 / (1.0 + parTMP.v * std::exp(-parTMP.s * 2 * (densityOfInfected - parTMP.h - 0.5)));
                // std::cout << y;
                y = parTMP.a * y + parTMP.b;
                // std::cout << ", " << y << '\n';
                return y / (60.0 * 24.0 / static_cast<double>(timeStep));
            });

        LocationsList<SimulationType>::infectAgents(infectionRatios, agentLocations);
    }
};
