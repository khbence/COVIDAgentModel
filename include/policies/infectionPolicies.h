#pragma once
#include "location.h"
#include "util.h"
#include <iostream>

template<class SimulationType>
class BasicInfection {
    double virulency = 0.0;
    double virulencyNorm = 5.0;
    double distance = 1.2;

public:
    BasicInfection(cxxopts::Options& options) {
        options.add_options()("virulency",
            "BasicInfection: virulency parameter",
            cxxopts::value<double>()->default_value("0.0"))("virulencyNorm",
            "BasicInfection: k=p/virulencyNorm",
            cxxopts::value<double>()->default_value("5.0"))("distance",
            "BasicInfection: used for densityOfInfected",
            cxxopts::value<double>()->default_value("1.2"));
    }

protected:
    void initialize_args(cxxopts::ParseResult& result) {
        virulency = result["virulency"].as<double>();
        virulencyNorm = result["virulencyNorm"].as<double>();
        distance = result["distance"].as<double>();
    }

public:
    void infectionsAtLocations(unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationListOffsets =
            realThis->locs
                ->locationListOffsets;// offsets into locationAgentList and locationIdsOfAgents
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;
        auto& ppstates = realThis->agents->PPValues;

        thrust::device_vector<double> infectionRatios(locationListOffsets.size() - 1, 0.0);
        thrust::device_vector<unsigned> fullInfectedCounts(locationListOffsets.size() - 1, 0);
        // Cout up infectious people - those who are Infected, and Infected state is >1
        reduce_by_location(locationListOffsets,
            fullInfectedCounts,
            ppstates,
            [] HD(typename SimulationType::PPState_t & ppstate) {
                return (unsigned)(ppstate.isInfectious());
            });
        double virulency2 = virulency;
        double virulencyNorm2 = virulencyNorm;
        double distance2 = distance;
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.begin(),
                              locationListOffsets.begin(),
                              locationListOffsets.begin() + 1)),
            thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.end(),
                locationListOffsets.end() - 1,
                locationListOffsets.end())),
            infectionRatios.begin(),
            [=] HD(thrust::tuple<unsigned&, unsigned&, unsigned&> tuple) {
                unsigned numInfectedAgentsPresent = thrust::get<0>(tuple);
                unsigned offset0 = thrust::get<1>(tuple);
                unsigned offset1 = thrust::get<2>(tuple);
                unsigned num_agents = offset1 - offset0;
                if (numInfectedAgentsPresent == 0) return 0.0;
                double densityOfInfected =
                    double(numInfectedAgentsPresent) / (num_agents * distance2);
                double p = 0.35 - virulency2;
                double k = p / virulencyNorm2;
                double y = 1.0 / (1.0 + exp((p - densityOfInfected) / k));
                return y / (60.0 * 24.0 / (double)timeStep);
            });

        LocationsList<SimulationType>::infectAgents(infectionRatios, agentLocations);
    }
};
