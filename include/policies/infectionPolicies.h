#pragma once
#include "locationList.h"
#include "util.h"
#include <iostream>

template<class SimulationType>
class BasicInfection {
    double virulency = 0.0;

    template<typename LocationType>
    double getInfectionRatio(LocationType& loc, unsigned timeStep) {

        auto& agents = loc.getAgents();
        auto realThis = static_cast<SimulationType*>(this);
        auto& ppstates = realThis->agents->PPValues;
        unsigned numInfectedAgentsPresent =
            thrust::count_if(make_permutation_iterator(ppstates.begin(), agents.begin()),
                make_permutation_iterator(ppstates.begin(), agents.end()),
                [](auto ppstate) { return ppstate.getSIRD() == states::SIRD::I; });
        unsigned total = agents.size();
        if (numInfectedAgentsPresent == 0) return 0;
        double densityOfInfected = double(numInfectedAgentsPresent) / (agents.size() * 1.2);
        double p = 0.35 - virulency;
        double k = p / 5.0;
        double y = 1.0 / (1.0 + exp((p - densityOfInfected) / k));
        unsigned timeStepCopy = static_cast<SimulationType*>(this)->timeStep;
        return y / (60.0 * 24.0 / (double)timeStep);
    }

protected:
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
        reduce_by_location(locationListOffsets, fullInfectedCounts, ppstates, [](auto ppstate) {
            return (unsigned)(ppstate.getSIRD() == states::SIRD::I);
        });
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.begin(),
                              locationListOffsets.begin(),
                              locationListOffsets.begin() + 1)),
            thrust::make_zip_iterator(thrust::make_tuple(fullInfectedCounts.end(),
                locationListOffsets.end() - 1,
                locationListOffsets.end())),
            infectionRatios.begin(),
            [=](auto tuple) {
                unsigned numInfectedAgentsPresent = thrust::get<0>(tuple);
                unsigned offset0 = thrust::get<1>(tuple);
                unsigned offset1 = thrust::get<2>(tuple);
                unsigned num_agents = offset1 - offset0;
                if (numInfectedAgentsPresent == 0) return 0.0;
                double densityOfInfected = double(numInfectedAgentsPresent) / (num_agents * 1.2);
                double p = 0.35 - virulency;
                double k = p / 5.0;
                double y = 1.0 / (1.0 + exp((p - densityOfInfected) / k));
                return y / (60.0 * 24.0 / (double)timeStep);
            });

        LocationsList<SimulationType>::infectAgents(infectionRatios, agentLocations);
    }
};