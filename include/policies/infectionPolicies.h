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
    unsigned flagInfectionAtLocations = 0;
    thrust::device_vector<unsigned> newInfectionsAtLocationsAccumulator;
    thrust::device_vector<bool> infectionFlagAtLocations;
    std::ofstream file;
    thrust::device_vector<unsigned> susceptible1;

public:
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("m,Imax", "Infection: [0 1] :max value ", cxxopts::value<double>()->default_value("0.32559"))("v,Iasymmetry",
            "Infection: 1< :longer small phase [0 1] :longer high phase",
            cxxopts::value<double>()->default_value("1.0"))("H,Ihorizontal",
            "sigmoid: horizotal move of inflexcion point; <-1 or >1 :exponential like",
            cxxopts::value<double>()->default_value("-0.42991"))("s,Ishape", "shape: bigger, more steep", cxxopts::value<double>()->default_value("22.25235"))(
            "dumpLocationInfections", "Dump per-location statistics every N timestep ", cxxopts::value<unsigned>()->default_value("0"))(
            "dumpLocationInfectiousList", "Dump per-location list of infectious people ", cxxopts::value<unsigned>()->default_value("0"));
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
        flagInfectionAtLocations = result["dumpLocationInfectiousList"].as<unsigned>();
    }

public:

    template <typename PPState_t>
    void dumpToFileStep1(thrust::device_vector<unsigned>& locationListOffsets,
                         thrust::device_vector<unsigned>& locationAgentList,
                         thrust::device_vector<PPState_t>& ppstates,
                         thrust::device_vector<float> &fullInfectedCounts,
                         thrust::device_vector<unsigned>& agentLocations,
                         Timehandler& simTime) {
        if (susceptible1.size()==0) {
            susceptible1.resize(locationListOffsets.size() - 1, 0);
        } else {
            thrust::fill(susceptible1.begin(), susceptible1.end(),0u);
        }
        
        if (newInfectionsAtLocationsAccumulator.size() == 0) {
            newInfectionsAtLocationsAccumulator.resize(locationListOffsets.size() - 1);
            thrust::fill(newInfectionsAtLocationsAccumulator.begin(), newInfectionsAtLocationsAccumulator.end(), (unsigned)0);
        }
        if (dumpToFile>0) { //Aggregate new infected counts
            reduce_by_location(locationListOffsets, locationAgentList, susceptible1, ppstates, agentLocations, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
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
            reduce_by_location(locationListOffsets, locationAgentList, infectedCount, ppstates, agentLocations, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
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
    }

    template <typename PPState_t>
    void dumpToFileStep2(thrust::device_vector<unsigned>& locationListOffsets,
                         thrust::device_vector<unsigned>& locationAgentList,
                         thrust::device_vector<PPState_t>& ppstates,
                         thrust::device_vector<unsigned>& agentLocations,
                         Timehandler& simTime) {
        if (dumpToFile>0) { //Finish aggregating number of new infections
            thrust::device_vector<unsigned> susceptible2(locationListOffsets.size() - 1, 0);
            reduce_by_location(locationListOffsets, locationAgentList, susceptible2, ppstates, agentLocations, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
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

    template <typename PPState_t>
    void dumpLocationInfectiousList(thrust::device_vector<PPState_t>& ppstates,
                                    thrust::device_vector<unsigned>& agentLocations,
                                    thrust::device_vector<unsigned>& fullInfectedCounts,
                                    Timehandler& simTime) {

        thrust::device_vector<unsigned> outLocationIdOffsets(infectionFlagAtLocations.size());
        auto b2u = []HD(bool flag)->unsigned {return flag?1u:0u;};
        thrust::exclusive_scan(thrust::make_transform_iterator(
                                    infectionFlagAtLocations.begin(),
                                    b2u),
                               thrust::make_transform_iterator(
                                    infectionFlagAtLocations.end(),
                                    b2u),
                               outLocationIdOffsets.begin());
        unsigned numberOfLocsWithInfections = outLocationIdOffsets[outLocationIdOffsets.size()-1]+
                                              infectionFlagAtLocations[infectionFlagAtLocations.size()-1];
        //no new infections inthis timestep, early exit
        if (numberOfLocsWithInfections == 0) return;

        //
        // List of location IDs
        //
        thrust::device_vector<unsigned> outLocationIds(numberOfLocsWithInfections);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                            infectionFlagAtLocations.begin(),
                            thrust::make_permutation_iterator(
                                outLocationIds.begin(),
                                outLocationIdOffsets.begin()),
                            thrust::make_counting_iterator<unsigned>(0))),
                         thrust::make_zip_iterator(thrust::make_tuple(
                            infectionFlagAtLocations.end(),
                            thrust::make_permutation_iterator(
                                outLocationIds.begin(),
                                outLocationIdOffsets.end()),
                            thrust::make_counting_iterator<unsigned>(0)+infectionFlagAtLocations.size())),
                         []HD(thrust::tuple<bool&, unsigned&, unsigned> tup) {
                             if (thrust::get<0>(tup)) //if infection at this loc
                                thrust::get<1>(tup) = thrust::get<2>(tup); //then save loc ID
                         });
        // std::cout << "list of locs ";
        // thrust::copy(outLocationIds.begin(), outLocationIds.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        // std::cout << "\n";
        
        //
        // Length of list at each location, scanned
        //
        // std::cout << "infected counts  ";
        // thrust::copy(fullInfectedCounts.begin(), fullInfectedCounts.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        // std::cout << "\n";
        // std::cout << "masks  ";
        // thrust::copy(infectionFlagAtLocations.begin(), infectionFlagAtLocations.end(), std::ostream_iterator<bool>(std::cout, " "));
        // std::cout << "\n";
        thrust::for_each(thrust::make_zip_iterator(
                            thrust::make_tuple(fullInfectedCounts.begin(),infectionFlagAtLocations.begin())),
                         thrust::make_zip_iterator(
                            thrust::make_tuple(fullInfectedCounts.end(),infectionFlagAtLocations.end())),
                            []HD(thrust::tuple<unsigned&, bool&> tup) {
                                if (thrust::get<1>(tup)==false) thrust::get<0>(tup) = 0;
                            });
        
        // std::cout << "infected counts masked with flag ";
        // thrust::copy(fullInfectedCounts.begin(), fullInfectedCounts.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        // std::cout << "\n";
        thrust::device_vector<unsigned> locationLengthAll(fullInfectedCounts.size(),0);
        thrust::exclusive_scan(fullInfectedCounts.begin(),
                               fullInfectedCounts.end(),
                               locationLengthAll.begin());
        thrust::device_vector<unsigned> locationLength(numberOfLocsWithInfections+1,0);
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                            infectionFlagAtLocations.begin(),
                            thrust::make_permutation_iterator(
                                locationLength.begin(),
                                outLocationIdOffsets.begin()),
                            locationLengthAll.begin())),
                         thrust::make_zip_iterator(thrust::make_tuple(
                            infectionFlagAtLocations.end(),
                            thrust::make_permutation_iterator(
                                locationLength.begin(),
                                outLocationIdOffsets.end()),
                            locationLengthAll.end())),
                         []HD(thrust::tuple<bool&, unsigned&, unsigned> tup) {
                             if (thrust::get<0>(tup)) //if infection at this loc
                                thrust::get<1>(tup) = thrust::get<2>(tup); //then offset
                         });
        locationLength[locationLength.size()-1] = locationLengthAll[locationLengthAll.size()-1]+
                                                    fullInfectedCounts[fullInfectedCounts.size()-1];
        // std::cout << "scanned location lengths ";
        // thrust::copy(locationLength.begin(), locationLength.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        // std::cout << "\n";
        //
        // indexes of people
        //
        thrust::device_vector<unsigned> peopleFlags(ppstates.size(),0u);
        thrust::for_each(thrust::make_zip_iterator(
                            thrust::make_tuple(ppstates.begin(),
                            peopleFlags.begin(),
                            thrust::make_permutation_iterator(
                                    infectionFlagAtLocations.begin(),
                                    agentLocations.begin()))),
                        thrust::make_zip_iterator(
                            thrust::make_tuple(ppstates.end(),
                            peopleFlags.end(),
                            thrust::make_permutation_iterator(
                                    infectionFlagAtLocations.begin(),
                                    agentLocations.end()))),
                        []HD(thrust::tuple<PPState_t&, unsigned&, bool&> tup) {
                            PPState_t& state = thrust::get<0>(tup);
                            unsigned& personFlag = thrust::get<1>(tup);
                            bool &locationFlag = thrust::get<2>(tup);
                            if (locationFlag && state.isInfectious())
                                personFlag = 1;
                        });
        // std::cout << "peopleFlags ";
        // thrust::copy(peopleFlags.begin(), peopleFlags.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        // std::cout << "\n";
        //rearrange people flags
        thrust::device_vector<unsigned> peopleFlagsByLocation(peopleFlags.size());
        thrust::device_vector<unsigned> peopleOffsetsByLocation(peopleFlags.size());
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationAgentList =
            realThis->locs->locationAgentList;
        thrust::copy(thrust::make_permutation_iterator(peopleFlags.begin(),locationAgentList.begin()),
                     thrust::make_permutation_iterator(peopleFlags.begin(),locationAgentList.end()),
                     peopleFlagsByLocation.begin());
        // std::cout << "peopleFlagsByLocation ";
        // thrust::copy(peopleFlagsByLocation.begin(), peopleFlagsByLocation.end(), std::ostream_iterator<unsigned>(std::cout, " "));
        // std::cout << "\n";
        thrust::exclusive_scan(peopleFlagsByLocation.begin(),peopleFlagsByLocation.end(),peopleOffsetsByLocation.begin());
        unsigned numberOfPeople = peopleOffsetsByLocation[peopleOffsetsByLocation.size()-1]+peopleFlagsByLocation[peopleFlagsByLocation.size()-1];
        if (numberOfPeople != locationLength[locationLength.size()-1]) { throw CustomErrors("dumpLocationInfectiousList: mismatch between number of people calculations"); }
        thrust::device_vector<unsigned> peopleIds(numberOfPeople);
        thrust::for_each(thrust::make_zip_iterator(
                            thrust::make_tuple(peopleFlagsByLocation.begin(),
                            locationAgentList.begin(),
                            thrust::make_permutation_iterator(
                                    peopleIds.begin(),
                                    peopleOffsetsByLocation.begin()))),
                        thrust::make_zip_iterator(
                            thrust::make_tuple(peopleFlagsByLocation.end(),
                            locationAgentList.end(),
                            thrust::make_permutation_iterator(
                                    peopleIds.begin(),
                                    peopleOffsetsByLocation.end()))),
                        []HD(thrust::tuple<unsigned&, unsigned&, unsigned&> tup) {
                            if (thrust::get<0>(tup)) //if person should be written
                                thrust::get<2>(tup) = thrust::get<1>(tup); //then write ID there
                        });
        std::ofstream file;
        file.open("infectiousList_" + std::to_string(simTime.getTimestamp()) + ".txt");
        file << numberOfLocsWithInfections << "\n";
        thrust::copy(outLocationIds.begin(), outLocationIds.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        thrust::copy(locationLength.begin(), locationLength.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        thrust::copy(peopleIds.begin(), peopleIds.end(), std::ostream_iterator<unsigned>(file, " "));
        file << "\n";
        file.close();
    }


    void infectionsAtLocations(Timehandler& simTime, unsigned timeStep) {
        PROFILE_FUNCTION();
        auto realThis = static_cast<SimulationType*>(this);
        thrust::device_vector<unsigned>& locationListOffsets =
            realThis->locs->locationListOffsets;// offsets into locationAgentList and locationIdsOfAgents
        thrust::device_vector<unsigned>& locationAgentList =
            realThis->locs->locationAgentList;
        thrust::device_vector<unsigned>& agentLocations = realThis->agents->location;

        auto& ppstates = realThis->agents->PPValues;
        auto& infectiousness = realThis->locs->infectiousness;

        thrust::device_vector<double> infectionRatios(locationListOffsets.size() - 1, 0.0);
        thrust::device_vector<float> fullInfectedCounts(locationListOffsets.size() - 1, 0);

        if (infectionFlagAtLocations.size()==0)
            infectionFlagAtLocations.resize(infectiousness.size());
        if (flagInfectionAtLocations)
            thrust::fill(infectionFlagAtLocations.begin(), infectionFlagAtLocations.end(), false);

        //
        // Step 1 - Count up infectious people - those who are Infectious
        //
        reduce_by_location(locationListOffsets, locationAgentList, fullInfectedCounts, ppstates, agentLocations, [] HD(const typename SimulationType::PPState_t& ppstate) -> float {
            return ppstate.isInfectious();
        });

        dumpToFileStep1(locationListOffsets, locationAgentList, ppstates, fullInfectedCounts, agentLocations, simTime);

        //
        // Step 2 - calculate infection ratios, based on density of infected people
        //
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

        //
        // Step 3 - randomly infect susceptible people
        //
        LocationsList<SimulationType>::infectAgents(infectionRatios, agentLocations, infectionFlagAtLocations, flagInfectionAtLocations, simTime);
        if (flagInfectionAtLocations) {
            thrust::device_vector<unsigned> fullInfectedCounts2(fullInfectedCounts.size(),0);
            reduce_by_location(locationListOffsets, locationAgentList, fullInfectedCounts2, ppstates, agentLocations, [] HD(const typename SimulationType::PPState_t& ppstate) -> unsigned {
                return unsigned(ppstate.isInfectious()>0);
            });
            dumpLocationInfectiousList(ppstates, agentLocations, fullInfectedCounts2, simTime);
        }

        dumpToFileStep2(locationListOffsets, locationAgentList, ppstates, agentLocations, simTime);
    }
};
