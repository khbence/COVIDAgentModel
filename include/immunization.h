#pragma once
#include <vector>
#include "datatypes.h"
#include <string>
#include "agentType.h"
#include <map>
#include "parametersFormat.h"
#include "agentTypesFormat.h"
#include "customExceptions.h"
#include "globalStates.h"
#include "timeHandler.h"
#include <iterator>
#include "agentsFormat.h"
#include "agentMeta.h"
#include "agentStats.h"
#include "agentStatOutput.h"
#include "progressionMatrixFormat.h"
#include "dataProvider.h"
#include "progressionType.h"


template <class Simulation>
class Immunization {
    Simulation *sim;
    thrust::device_vector<uint8_t> immunizationRound;
    unsigned currentCategory = 0;
    unsigned numberOfCategories = 0;
    unsigned startAfterDay = 0;
    unsigned dailyDoses = 0;

    unsigned numberOfVaccinesToday(Timehandler& simTime) {
        if (simTime.getTimestamp()/(24*60/simTime.getTimeStep()) >= startAfterDay) return dailyDoses;
        else return 0;
    }

    public:
    Immunization (Simulation *s) {
        this->sim = s;
    }
    static void addProgramParameters(cxxopts::Options& options) {
        options.add_options()("immunizationStart",
            "number of days into simulation when immunization starts",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))))
            ("immunizationsPerDay",
            "number of immunizations per day",
            cxxopts::value<unsigned>()->default_value(std::to_string(unsigned(0))));
    }

    void initializeArgs(const cxxopts::ParseResult& result) {
        startAfterDay = result["immunizationStart"].as<unsigned>();
        dailyDoses = result["immunizationsPerDay"].as<unsigned>();
    }

    void initCategories() {
        immunizationRound.resize(sim->agents->PPValues.size(),0);
        
        auto *agentMetaDataPtr = thrust::raw_pointer_cast(sim->agents->agentMetaData.data());
        auto *locationOffsetPtr = thrust::raw_pointer_cast(sim->agents->locationOffset.data());
        auto *possibleTypesPtr = thrust::raw_pointer_cast(sim->agents->possibleTypes.data());
        auto *locationTypePtr = thrust::raw_pointer_cast(sim->locs->locType.data());
        auto *possibleLocationsPtr = thrust::raw_pointer_cast(sim->agents->possibleLocations.data());
        auto *essentialPtr = thrust::raw_pointer_cast(sim->locs->essential.data());
        

        //Figure out which category agents belong to, and determine if agent is willing to be vaccinated

        //Category health worker
        auto cat_healthworker = [locationOffsetPtr, possibleTypesPtr,possibleLocationsPtr,locationTypePtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                //TODO pull these params from config
                if (possibleTypesPtr[idx] == 4 && (locationTypePtr[possibleLocationsPtr[idx]]==12 || locationTypePtr[possibleLocationsPtr[idx]]==14))
                    return thrust::make_pair(true, 0.7f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category nursery home workers & residents
        auto cat_nursery = [locationOffsetPtr, possibleTypesPtr,locationTypePtr,possibleLocationsPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                if ((possibleTypesPtr[idx] == 4 || possibleTypesPtr[idx] == 2) && locationTypePtr[possibleLocationsPtr[idx]]==22) //TODO pull these params from config
                    return thrust::make_pair(true, 0.9f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category elderly
        auto cat_elderly = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getAge()>=60) return thrust::make_pair(true, 0.7f);
            else return thrust::make_pair(false,0.0f);
        };

        //Category 18-59, underlying condition
        auto cat_underlying = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getPrecondIdx()>0 && agentMetaDataPtr[id].getAge()>=18 && agentMetaDataPtr[id].getAge()<60) return thrust::make_pair(true, 0.8f);
            else return thrust::make_pair(false,0.0f);
        };

        //Category essential workers
        auto cat_essential = [locationOffsetPtr, possibleTypesPtr,essentialPtr,possibleLocationsPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            for (unsigned idx = locationOffsetPtr[id]; idx < locationOffsetPtr[id+1]; idx++) {
                if ((possibleTypesPtr[idx] == 4 || possibleTypesPtr[idx] == 2) && essentialPtr[possibleLocationsPtr[idx]]==1) //TODO pull these params from config
                    return thrust::make_pair(true, 0.9f);
            }
            return thrust::make_pair(false,0.0f);
        };

        //Category over 18
        auto cat_adult = [agentMetaDataPtr] HD (unsigned id) -> thrust::pair<bool,float> {
            if (agentMetaDataPtr[id].getAge()>17) return thrust::make_pair(true, 0.6f);
            else return thrust::make_pair(false,0.0f);
        };

        numberOfCategories = 6;

        //Figure out which round of immunizations agent belongs to, and decide if agent wants to get it.
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), thrust::make_counting_iterator(0))),
                         thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.end()  , thrust::make_counting_iterator((int)immunizationRound.size()))),
                            [cat_healthworker,cat_nursery,cat_elderly,cat_underlying,cat_essential,cat_adult] HD (thrust::tuple<uint8_t&, int> tup) {
                                uint8_t& round = thrust::get<0>(tup);
                                unsigned id = thrust::get<1>(tup);

                                uint8_t roundidx = 1;
                                auto ret = cat_healthworker(id);
                                if (ret.first && round == 0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = roundidx;
                                    else round = (uint8_t)-1;
                                    return;
                                }
                                roundidx++;

                                ret = cat_nursery(id);
                                if (ret.first && round == 0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = roundidx;
                                    else round = (uint8_t)-1;
                                    return;
                                }
                                roundidx++;

                                ret = cat_elderly(id);
                                if (ret.first && round == 0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = roundidx;
                                    else round = (uint8_t)-1;
                                    return;
                                }
                                roundidx++;

                                ret = cat_underlying(id);
                                if (ret.first && round == 0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = roundidx;
                                    else round = (uint8_t)-1;
                                    return;
                                }
                                roundidx++;

                                ret = cat_essential(id);
                                if (ret.first && round == 0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = roundidx;
                                    else round = (uint8_t)-1;
                                    return;
                                }
                                roundidx++;

                                ret = cat_adult(id);
                                if (ret.first && round == 0) {
                                    if (RandomGenerator::randomUnit() < ret.second) round = roundidx;
                                    else round = (uint8_t)-1;
                                    return;
                                }
                                roundidx++;
                            }
                        );
    }
    void update(Timehandler& simTime, unsigned timeStep) {
        unsigned timestamp = simTime.getTimestamp();
        if (timestamp == 0) timestamp++; //First day already immunizing, then we sohuld not set immunizedTimestamp to 0

        //Update immunity based on days since immunization
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(sim->agents->PPValues.begin(), sim->agents->agentStats.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(sim->agents->PPValues.end(), sim->agents->agentStats.end())),
                             [timeStep,timestamp]HD(thrust::tuple<typename Simulation::PPState_t&, AgentStats&> tup) {
                                 //If not immunized, or already recovered, return
                                 if (thrust::get<1>(tup).immunizationTimestamp == 0 || thrust::get<0>(tup).getSusceptible()==0.0) return;
                                 //Otherwise get more immune after days since immunization
                                 unsigned daysSinceImmunization = (timestamp-thrust::get<1>(tup).immunizationTimestamp)/(24*60/timeStep);
                                 if (daysSinceImmunization>=28) thrust::get<0>(tup).setSusceptible(0.04); //96%
                                 else if (daysSinceImmunization>=12) thrust::get<0>(tup).setSusceptible(0.48); //52%
                             });

        //no vaccines today, or everybody already immunized, return
        if (numberOfVaccinesToday(simTime) == 0 || currentCategory >= numberOfCategories) return;

        unsigned available = numberOfVaccinesToday(simTime);
        while (available > 0 && currentCategory < numberOfCategories) {
            //Count number of eligible in current group
            unsigned count = 0;
            while (count == 0 && currentCategory < numberOfCategories) {
                unsigned currentCategoryLocal = currentCategory+1; //agents' categories start at 1
                count = thrust::count_if(thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), sim->agents->agentStats.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.end(), sim->agents->agentStats.end())),
                             [currentCategoryLocal,timeStep,timestamp]HD(thrust::tuple<uint8_t, AgentStats> tup) {
                                 if (thrust::get<0>(tup) == currentCategoryLocal &&  //TODO how many days since diagnosis?
                                     thrust::get<1>(tup).immunizationTimestamp == 0 &&
                                     ((timestamp < (24*60/timeStep)*6*30 && thrust::get<1>(tup).diagnosedTimestamp == 0) ||
                                      (timestamp >= (24*60/timeStep)*6*30 && thrust::get<1>(tup).diagnosedTimestamp < timestamp - (24*60/timeStep)*6*30))) {
                                          return true;
                                      }
                                 return false;
                             });
                if (count == 0) currentCategory++;
            }
            

            //Probability of each getting vaccinated today
            float prob;
            if (count < available) prob = 1.0;
            else prob = (float)available/(float)count;
            //printf("count %d avail %d category %d\n", count, available, currentCategory);

            //immunize available number of people in current category
            unsigned currentCategoryLocal = currentCategory+1;
            thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.begin(), sim->agents->agentStats.begin())),
                             thrust::make_zip_iterator(thrust::make_tuple(immunizationRound.end(), sim->agents->agentStats.end())),
                             [currentCategoryLocal,timeStep,timestamp,prob]HD(thrust::tuple<uint8_t&, AgentStats&> tup) {
                                 if (thrust::get<0>(tup) == currentCategoryLocal &&  //TODO how many days since diagnosis?
                                     thrust::get<1>(tup).immunizationTimestamp == 0 &&
                                     ((timestamp < (24*60/timeStep)*6*30 && thrust::get<1>(tup).diagnosedTimestamp == 0) ||
                                      (timestamp >= (24*60/timeStep)*6*30 && thrust::get<1>(tup).diagnosedTimestamp < timestamp - (24*60/timeStep)*6*30))) {
                                          if (prob == 1.0f || RandomGenerator::randomUnit() < prob)
                                          thrust::get<1>(tup).immunizationTimestamp = timestamp;
                                      }
                             });

            //subtract from available
            if (count < available) available -= count;
            else available = 0;
        }
    }
};