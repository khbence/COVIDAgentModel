#pragma once
#include "locationList.h"
#include "util.h"
#include "timeHandler.h"
#include <iostream>

template<class SimulationType>
class BasicInfection {
public:
    struct Parameters {
        double v, h, s;
        double a, b;

        Parameters() = default;
    };

private:
    Parameters par;
    unsigned dumpToFile = 0;
    thrust::device_vector<unsigned> newInfectionsAtLocationsAccumulator;

public:
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("m,Imax", "Infection: [0 1] :max value ", cxxopts::value<double>()->default_value("1.0"))("v,Iasymmetry",
            "Infection: 1< :longer small phase [0 1] :longer high phase",
            cxxopts::value<double>()->default_value("1.0"))("H,Ihorizontal",
            "sigmoid: horizotal move of inflexcion point; <-1 or >1 :exponential like",
            cxxopts::value<double>()->default_value("0.0"))("s,Ishape", "shape: bigger, more steep", cxxopts::value<double>()->default_value("5.0"))(
            "dumpLocationInfections", "Dump per-location statistics every N timestep ", cxxopts::value<unsigned>()->default_value("0"));
    }

protected:
    void initializeArgs(const cxxopts::ParseResult& result) {
        auto m = result["Imax"].as<double>() / 2;
        par.v = result["Iasymmetry"].as<double>();
        par.h = result["Ihorizontal"].as<double>();
        par.s = result["Ishape"].as<double>();
        double min = 1.0 / (1.0 + par.v * std::exp(-par.s * 2 * (0.0 - par.h - 0.5)));
        double max = 1.0 / (1.0 + par.v * std::exp(-par.s * 2 * (1.0 - par.h - 0.5)));

        par.a = m / (max - min);
        par.b = -m * min / (max - min);
        dumpToFile = result["dumpLocationInfections"].as<unsigned>();
    }

public:
    void infectionsAtLocations(Timehandler& simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationListOffsets =
            realThis->locs->locationListOffsets;// offsets into locationAgentList and locationIdsOfAgents
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;

        auto& ppstates = realThis->agents->PPValues;
        auto& infectiousness = realThis->locs->infectiousness;

        thrust::device_vector<double> infectionRatios(locationListOffsets.size() - 1, 0.0);
        thrust::device_vector<float> fullInfectedCounts(locationListOffsets.size() - 1, 0);
        // Count up infectious people - those who are Infected, and Infected state is >1
        reduce_by_location(locationListOffsets, fullInfectedCounts, ppstates, [] HD(const typename SimulationType::PPState_t& ppstate) -> float {
            return ppstate.isInfectious();
        });
        std::ofstream file;
        if (newInfectionsAtLocationsAccumulator.size() == 0) {
            newInfectionsAtLocationsAccumulator.resize(locationListOffsets.size() - 1);
            thrust::fill(newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.end(), (unsigned)0);
        }
        thrust::device_vector<unsigned> susceptible1;
        if (dumpToFile>0) { //Aggregate new infected counts
            susceptible1.resize(locationListOffsets.size() - 1, 0);
            reduce_by_location(locationListOffsets, susceptible1, ppstates, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
                return ppstate.isSusceptible();
            });
        }
        if (dumpToFile>0 && simTime.getTimestamp()%dumpToFile == 0) {
            file.open("locationStats_" + std::to_string(simTime.getTimestamp()) + ".txt");
            //Count number of people at each location
            thrust::device_vector<unsigned> location(locationListOffsets.size() - 1);
            thrust::transform(
                locationListOffsets.begin() + 1, locationListOffsets.end(), locationListOffsets.begin(), location.begin(), thrust::minus<unsigned>());
            //Count number of infected people at each locaiton
            thrust::device_vector<unsigned> infectedCount(locationListOffsets.size() - 1, 0);
            reduce_by_location(locationListOffsets, infectedCount, ppstates, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
                return ppstate.isInfectious() > 0;
            });
            //Print people/location
            thrust::copy(location.begin(), location.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            //Print infected/location
            thrust::copy(infectedCount.begin(), infectedCount.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            //Print weighted infected counts too
            thrust::copy(fullInfectedCounts.begin(), fullInfectedCounts.end(), std::ostream_iterator<float>(file, " "));
            file << "\n";
        }
        auto parTMP = par;
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                              fullInfectedCounts.begin(), locationListOffsets.begin(), locationListOffsets.begin() + 1, infectiousness.begin())),
            thrust::make_zip_iterator(
                thrust::make_tuple(fullInfectedCounts.end(), locationListOffsets.end() - 1, locationListOffsets.end(), infectiousness.end())),
            infectionRatios.begin(),
            [=] HD(thrust::tuple<float&, unsigned&, unsigned&, double&> tuple) {
                float numInfectedAgentsPresent = thrust::get<0>(tuple);
                unsigned offset0 = thrust::get<1>(tuple);
                unsigned offset1 = thrust::get<2>(tuple);
                unsigned num_agents = offset1 - offset0;
                if (numInfectedAgentsPresent == 0.0) { return 0.0; }
                double densityOfInfected = numInfectedAgentsPresent / num_agents;
                double y = 1.0 / (1.0 + parTMP.v * std::exp(-parTMP.s * 2 * (densityOfInfected - parTMP.h - 0.5)));
                y = parTMP.a * y + parTMP.b;
                y *= thrust::get<3>(tuple);// Weighted by infectiousness
                return y / (60.0 * 24.0 / static_cast<double>(timeStep));
            });

        LocationsList<SimulationType>::infectAgents(infectionRatios, agentLocations, simTime);
        if (dumpToFile>0) { //Finish aggregating number of new infections
            thrust::device_vector<unsigned> susceptible2(locationListOffsets.size() - 1, 0);
            reduce_by_location(locationListOffsets, susceptible2, ppstates, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
                return ppstate.isSusceptible();
            });
            thrust::transform(susceptible1.begin(), susceptible1.end(), susceptible2.begin(), susceptible1.begin(), thrust::minus<unsigned>());
            thrust::transform(susceptible1.begin(), susceptible1.end(), newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.begin(), thrust::plus<unsigned>());
        }
        if (dumpToFile>0 && simTime.getTimestamp()%dumpToFile==0) {
            //Print new infections at location since last timestep
            thrust::copy(newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.end(), std::ostream_iterator<unsigned>(file, " "));
            file << "\n";
            file.close();
            thrust::fill(newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.end(), (unsigned)0);
        }
    }
};
